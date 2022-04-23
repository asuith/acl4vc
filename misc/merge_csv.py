import argparse
import glob
import os
from pprint import pprint

import pandas
import torch

pandas.set_option("display.precision", 2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset", default="MSRVTT", choices=['MSVD', 'MSRVTT', 'VATEX', 'myV'])
    parser.add_argument("-e", "--exp", default="experiments", choices=['experiments', 'Ae', ])
    parser.add_argument("-ss", "--skip_scopes", nargs='+', type=str, default=['test_'],
                        help="skip scopes whose names are exact one of these")
    parser.add_argument("-sm", "--skip_models", nargs='+', type=str, default=['Archived',
                                                                              # "ARB_ami_CL",
                                                                              ])
    parser.add_argument("--only_name", type=str, default=None)


    # "Transformer", "Transformer_m8"])
    parser.add_argument("--remove_columns", nargs='+', type=str, default=None,
                        help="skip columns whose names are exact one of these")
    parser.add_argument("--sorted_by", nargs='+', type=str, default=["model_name", "Sum", ],
                        help="place model_name first")

    # Bleu_1 Bleu_2 Bleu_3 Sum
    parser.add_argument("-name", "--output_name", type=str, default="merged_all_csv",
                        help="output file name.")
    parser.add_argument("--csv_name", type=str, default="test_result.csv", )
    parser.add_argument("--load_extra", default=False, action="store_true")
    parser.add_argument("--round", type=int, default=2, help="round decimals to this number of digits")

    args = parser.parse_args()

    BASE_PATH = os.path.join(args.exp, args.dataset)
    # find path
    path = os.path.join(BASE_PATH, f"*/*/{args.csv_name}")
    print(path)
    models_paths = glob.glob(path)
    models_paths = sorted(models_paths)

    # skip some file
    new_paths = []
    for path in models_paths:
        ps = path.split("/")
        #         print(ps)
        #         print(ps[3])
        #         break
        model_name = ps[2]
        scope_name = ps[3]
        if model_name in args.skip_models:
            continue
        if scope_name in args.skip_scopes:
            continue
        new_paths.append(path)
    if len(args.only_name) is not None:
        new_paths = [p for p in new_paths if args.only_name.lower() in p.lower()]
    models_paths = new_paths

    # merge
    csv_data = []
    for i, path in enumerate(models_paths):
        ps = path.split("/")
        model_name = ps[2]
        scope_name = ps[3]
        csv_df = pandas.read_csv(path)
        csv_df.insert(0, "model_name", model_name)
        csv_df.insert(1, "scope_name", scope_name)
        if args.load_extra:
            model_path = path.replace("test_result.csv", "best.ckpt")
            # print(model_path)
            model = torch.load(model_path)
            epoch = model["epoch"]
            global_step = model["global_step"]
            del model
            csv_df.insert(len(csv_df.columns), "global_step", global_step)
            csv_df.insert(len(csv_df.columns), "epoch", epoch)
        csv_data.append(csv_df)
        # print(csv_df)
    assert len(csv_data) > 0, f"No test data in `{args.exp}` dir for dataset `{args.dataset}`"
    # all_df = pandas.concat(csv_data).sort_values(["model_name", "scope_name",])
    all_df = pandas.concat(csv_data).sort_values(args.sorted_by)
    # to percentage and .2 precision
    # all_df[all_df.select_dtypes(include=['number']).columns] *= 100
    all_df[["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "Sum"]] *= 100
    all_df = all_df.round(args.round)
    # all_df[all_df.select_dtypes(include=['number']).columns].map(lambda x: '%.2f' % x)
    if args.remove_columns:
        all_df = all_df.drop(args.remove_columns, axis=1)
    rename_dict = {
        "Bleu_4": "BLEU@4",
        "ROUGE_L": "ROUGE-L",
    }
    all_df.rename(columns=rename_dict, inplace=True)
    pprint(all_df)
    # finalnp = finaldf.to_numpy()
    # output
    output_file_name = args.dataset + "-" + (args.output_name if ".csv" in args.output_name else args.output_name + ".csv")
    output_path = os.path.join(BASE_PATH, f"{output_file_name}")
    print("Output to:", output_path)
    all_df.to_csv(output_path, index=False)

