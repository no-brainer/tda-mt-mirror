import argparse

import torch

from src.models import NMTTransformer
import src.tokenizers
import src.translators
from src.utils import parse_config, set_seed, init_obj


def main(args):
    training_config = parse_config(args.config_path)

    device = "cpu"
    if torch.cuda.is_available() and args.use_cuda:
        device = "cuda"
    print("Inference on", device)

    set_seed(training_config["seed"])

    saved_data = torch.load(args.checkpoint_path, map_location="cpu")
    model = NMTTransformer(**training_config["model"])
    model.load_state_dict(saved_data["state_dict"])
    model = model.to(device)

    tokenizer = init_obj(src.tokenizers, training_config["tokenizer"])
    if args.translator_type == "greedy":
        translator = src.translators.GreedyTranslator(model, tokenizer, device, args.bos_id, args.eos_id,
                                                      args.max_length)
    elif args.translator_type == "beam_search":
        translator = src.translators.BeamSearchTranslator(model, tokenizer, device, args.bos_id, args.eos_id,
                                                          args.max_length, args.beam_size, args.temperature)
    else:
        raise ValueError(f"Invalid translator type: {args.translator_type}")

    with open(args.src_datapath, "r") as in_file, \
            open(args.out_datapath, "w") as out_file:
        for line in in_file:
            line = line.strip()
            translation = translator.translate(line)
            out_file.write(translation)
            out_file.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("src_datapath", type=str)
    parser.add_argument("out_datapath", type=str)
    parser.add_argument("translator_type", choices=["greedy", "beam_search"])

    parser.add_argument("--config_path", "-c", type=str, required=True)
    parser.add_argument("--not_use_cuda", dest="use_cuda", action="store_false")

    parser.add_argument("--max_length", "-l", type=int, default=512)
    parser.add_argument("--bos_id", "-b", type=int, default=2)
    parser.add_argument("--eos_id", "-e", type=int, default=3)
    parser.add_argument("--beam_size", "-b", type=int, default=5)
    parser.add_argument("--temperature", "-t", type=float, default=1.)

    script_args = parser.parse_args()

    main(script_args)
