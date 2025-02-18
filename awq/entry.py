from lm_eval import evaluator
from lm_eval.utils import make_table
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import argparse
import os
import json
from accelerate import (
    init_empty_weights,
    infer_auto_device_map,
    dispatch_model,
    load_checkpoint_in_model,
)
from accelerate.utils.modeling import get_balanced_memory
from awq.quantize.qmodule import WQLinear
from awq.utils.parallel import auto_parallel
from awq.quantize.pre_quant import get_blocks, run_awq, apply_awq
from awq.quantize.quantizer import (
    pseudo_quantize_model_weight,
    get_scales_zeros,
    real_quantize_model_weight,
)
from lm_eval.models.huggingface import HFLM
from awq.utils.utils import simple_dispatch_model
from awq.scale_list_analysis.modify_scales import round_nearest_power_of_2, set_all_ones
from awq.scale_list_analysis.get_real_scales_and_zeros import export_scales_zeros
from datasets import load_dataset
from torch import nn
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help="path of the hf model")
parser.add_argument("--cache_dir", default="", type=str )
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--tasks", default=None, type=str)
parser.add_argument("--output_path", default=None, type=str)
parser.add_argument("--num_fewshot", type=int, default=0)
# model config
parser.add_argument("--parallel", action="store_true", help="enable model parallelism")
# max memory to offload larger models to CPU
parser.add_argument(
    "--max_memory",
    type=str,
    nargs="*",
    help="List of device_id:max_memory pairs to be parsed into a dictionary; "
    + "Example: 0:10GiB 1:10GiB cpu:30GiB; "
    + "mode details here: "
    + "https://huggingface.co/docs/accelerate/usage_guides/big_modeling",
)
parser.add_argument(
    "--auto_parallel",
    action="store_true",
    help="automatically set parallel and batch_size",
)
# quantization config
parser.add_argument("--w_bit", type=int, default=None)
parser.add_argument("--q_group_size", type=int, default=-1)
parser.add_argument("--threshold", type=float, default=0.0, help="threshold to set scale values to 1 (none if 0)")
parser.add_argument("--no_zero_point", action="store_true", help="disable zero_point")
parser.add_argument("--q_backend", type=str, default="fake", choices=["fake", "real"])
# save/load real quantized weights
parser.add_argument("--dump_quant", type=str, default=None, help="save quantized model")
parser.add_argument("--dump_fake", type=str, default=None, help="save fake-quantized model")
parser.add_argument("--dump_zero_scales", type=str, default=None, help="save fake-quantized model")
parser.add_argument("--load_quant", type=str, default=None, help="load quantized model")
# apply/save/load awq
parser.add_argument("--run_awq", action="store_true", help="perform awq search process")
parser.add_argument("--check_sparsity", action="store_true", help="check sparsity result of awq")
parser.add_argument("--round_to_p2", action="store_true", help="rounds awq scales to nearest power of 2")
parser.add_argument("--set_to_1", action="store_true", help="sets awq scales to 1 (removes awq)")
parser.add_argument("--prune_highbit", action="store_true", help="prunes at w=2 then sets non-pruned at w=4")
parser.add_argument("--eval_seqlen", type=int, default=2048)
parser.add_argument(
    "--dump_awq", type=str, default=None, help="save the awq search results"
)
parser.add_argument(
    "--load_awq", type=str, default=None, help="load the awq search results"
)
parser.add_argument(
    "--vila-15",
    action="store_true",
    help="quantizing vila 1.5",
)
args = parser.parse_args()
vila_10_quant_mode = ("llava" in args.model_path.lower() or "vila" in args.model_path.lower()) and not args.vila_15

max_memory = [v.split(":") for v in (args.max_memory or [])]
max_memory = {(int(k) if k.isdigit() else k): v for k, v in max_memory}

if args.auto_parallel:
    gpu_list = auto_parallel(args)

# get quantization config (apart from w_bit)
q_config = {
    "zero_point": not args.no_zero_point,  # by default True
    "q_group_size": args.q_group_size,  # whether to use group quantization
}
print("Quantization config:", q_config)

# build model and tokenizer


def build_model_and_enc(model_path):
    # if not os.path.exists(model_path):  # look into ssd
    #     raise FileNotFoundError(f"{model_path} not found!")
    print(f"* Building model {model_path}")

    # all hf model
    if vila_10_quant_mode:
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path

        enc, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path),
            device="cpu",
            **{"use_cache": False}
        )
    else:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        # Note (Haotian): To avoid OOM after huggingface transformers 4.36.2
        config.use_cache = False
        if "mpt" in config.__class__.__name__.lower():
            enc = AutoTokenizer.from_pretrained(
                config.tokenizer_name, trust_remote_code=True
            )
        else:
            enc = AutoTokenizer.from_pretrained(
                model_path, use_fast=False, trust_remote_code=True
            )

    if args.load_quant:  # directly load quantized weights
        print("Loading pre-computed quantized weights...")
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                config=config, torch_dtype=torch.float16, trust_remote_code=True
            )
        real_quantize_model_weight(
            model, w_bit=args.w_bit, q_config=q_config, init_only=True
        )

        model.tie_weights()

        # Infer device map
        kwargs = {"max_memory": max_memory} if len(max_memory) else {}
        device_map = infer_auto_device_map(
            model,
            no_split_module_classes=[
                "OPTDecoderLayer",
                "LlamaDecoderLayer",
                "BloomBlock",
                "MPTBlock",
                "DecoderLayer",
            ],
            **kwargs,
        )
        # Load checkpoint in the model
        load_checkpoint_in_model(
            model,
            checkpoint=args.load_quant,
            device_map=device_map,
            offload_state_dict=True,
        )
        # Dispatch model
        model = simple_dispatch_model(model, device_map=device_map)

        model.eval()
    else:  # fp16 to quantized
        args.run_awq &= not args.load_awq  # if load_awq, no need to run awq
        # Init model on CPU:
        kwargs = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}
        if not vila_10_quant_mode:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, cache_dir=args.cache_dir, config=config, trust_remote_code=True, **kwargs
            )
            if args.prune_highbit:
                model2 = AutoModelForCausalLM.from_pretrained(
                    model_path, cache_dir=args.cache_dir, config=config, trust_remote_code=True, **kwargs
                )
                model2.eval()

        model.eval()

        if args.run_awq:
            assert args.dump_awq, "Please save the awq results with --dump_awq"

            awq_results = run_awq(
                model,
                enc,
                w_bit=args.w_bit,
                q_config=q_config,
                n_samples=128,
                seqlen=512,
                threshold=args.threshold
            )
            if args.dump_awq:
                dirpath = os.path.dirname(args.dump_awq)
                os.makedirs(dirpath, exist_ok=True)

                torch.save(awq_results, args.dump_awq)
                print("AWQ results saved at", args.dump_awq)

            exit(0)

        if args.load_awq:
            print("Loading pre-computed AWQ results from", args.load_awq)
            awq_results = torch.load(args.load_awq, map_location="cpu")
            awq_results2 = torch.load(args.load_awq + "4", map_location="cpu")
            if args.round_to_p2:
                round_nearest_power_of_2(awq_results)
            if args.set_to_1:
                set_all_ones(awq_results)
            apply_awq(model, awq_results)
            if args.prune_highbit:
                apply_awq(model2, awq_results2)

        # weight quantization
        if args.w_bit is not None:
            if args.dump_zero_scales:
                scales, zeros = get_scales_zeros(model, w_bit=args.w_bit, q_config=q_config)
                export_scales_zeros(scales, zeros, os.path.join(args.dump_zero_scales, f'fake_scales'), os.path.join(args.dump_zero_scales, f'fake_zeros'))
                
            if args.q_backend == "fake":
                assert (
                    args.dump_quant is None
                ), "Need to use real quantization to dump quantized weights"
                pseudo_quantize_model_weight(model, w_bit=args.w_bit, q_config=q_config)

                if args.prune_highbit:
                    pseudo_quantize_model_weight(model2, w_bit=4, q_config=q_config)

                if args.dump_fake:
                    model.save_pretrained(args.dump_fake)
                    print("Pseudo-quantized models saved at", args.dump_fake)
            elif args.q_backend == "real":  # real quantization
                real_quantize_model_weight(model, w_bit=args.w_bit, q_config=q_config)
                if args.dump_quant:
                    if not args.dump_quant.endswith("v2.pt"):
                        print("[Info] Auto-change the dump_quant file name to *v2.pt")
                        args.dump_quant = args.dump_quant.replace(".pt", "-v2.pt")
                    dirpath = os.path.dirname(args.dump_quant)
                    os.makedirs(dirpath, exist_ok=True)

                    print(f"Saving the quantized model at {args.dump_quant}...")
                    torch.save(model.cpu().state_dict(), args.dump_quant)
                    exit(0)
            else:
                raise NotImplementedError

        # Move the model to GPU (as much as possible) for LM evaluation
        kwargs = {
            "max_memory": get_balanced_memory(
                model, max_memory if len(max_memory) > 0 else None
            )
        }
        device_map = infer_auto_device_map(
            model,
            # TODO: can we remove this?
            no_split_module_classes=[
                "OPTDecoderLayer",
                "LlamaDecoderLayer",
                "BloomBlock",
                "MPTBlock",
                "DecoderLayer",
            ],
            **kwargs,
        )
        model = dispatch_model(model, device_map=device_map)

    if args.prune_highbit:
        return model, model2, enc
    return model, enc


def main():
    if args.output_path is not None and os.path.exists(args.output_path):
        # print(f"Results {args.output_path} already generated. Exit.")
        print(f"Results {args.output_path} already generated. Overwrite.")
        # exit()

    if args.dump_awq and os.path.exists(args.dump_awq):
        print(f"Found existing AWQ results {args.dump_awq}, exit.")
        exit()

    # a hack here to auto set model group
    if args.prune_highbit:
        model, model2, enc = build_model_and_enc(args.model_path)
    else:
        model, enc = build_model_and_enc(args.model_path)

    if args.check_sparsity or args.prune_highbit:
        model = model.eval()
        layers = get_blocks(model)
        if args.prune_highbit:
            layers2 = get_blocks(model2)
        total_zeroes = torch.zeros(1)
        total_n = torch.zeros(1)
        for i in tqdm.tqdm(range(len(layers)), desc="checking sparsities..."):
            layer = layers[i]
            named_linears = {name: m for name, m in layer.named_modules() if isinstance(m, WQLinear) or isinstance(m, nn.Linear)}
            if args.prune_highbit:
                named_linears2 = {name: m for name, m in layers2[i].named_modules() if isinstance(m, WQLinear) or isinstance(m, nn.Linear)}
            for name, module in named_linears.items():
                if args.prune_highbit:
                    module.weight.data[module.weight.data != 0] = named_linears2[name].weight.data.to(module.weight.data.device)[module.weight.data != 0]
                if isinstance(module, WQLinear):
                    zeroes = torch.sum(module.qweight.data == 0).item()
                    n = module.qweight.data.numel()
                else:
                    zeroes = torch.sum(module.weight.data == 0).item()
                    n = module.weight.data.numel()
                print(f'layer {i} module {name} sparsity: {zeroes}/{n} {zeroes/n if n != 0 else 0}')
                total_zeroes += zeroes
                total_n += n
        print(f'total sparsity: {total_zeroes/total_n}')

    if args.tasks is not None:
        if args.tasks != 'wikitext':
            task_names = [x for x in args.tasks.split(",") if x != 'wikitext']

            lm = HFLM(model, max_length=args.eval_seqlen, parallelize=False)
            # lm_eval_model = LMEvalAdaptor(args.model_path, model, enc, args.batch_size)
            results = evaluator.simple_evaluate(
                model=lm,
                tasks=task_names,
                batch_size=args.batch_size,
                num_fewshot=args.num_fewshot,
            )

            # print(results)
            print(make_table(results))

        # https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/llama.py#L206
        if 'wikitext' in args.tasks:
            testenc = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            testenc = enc("\n\n".join(testenc["text"]), return_tensors="pt")
            model.seqlen = args.eval_seqlen
            testenc = testenc.input_ids.to(model.device)
            nsamples = testenc.numel() // model.seqlen
            model = model.eval()
            nlls = []
            for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
                batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(
                    model.device
                )
                with torch.no_grad():
                    lm_logits = model(batch).logits
                shift_logits = lm_logits[:, :-1, :].contiguous().float()
                shift_labels = testenc[
                    :, (i * model.seqlen) : ((i + 1) * model.seqlen)
                ][:, 1:]
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )
                neg_log_likelihood = loss.float() * model.seqlen
                nlls.append(neg_log_likelihood)

            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
            print(ppl.item())

            results = {"ppl": ppl.item()}
            if args.output_path is not None:
                os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
                with open(args.output_path, "w") as f:
                    json.dump(results, f, indent=2)

        if args.output_path is not None:
            os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
            # otherwise cannot save
            results["config"]["model"] = args.model_path
            with open(args.output_path, "w") as f:
                json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
