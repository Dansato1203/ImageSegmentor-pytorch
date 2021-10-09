#! usr/bin/env python3

import argparse
import base64
import json
import os
import os.path as osp
import glob
import datetime

import imgviz
import PIL.Image

from labelme.logger import logger
from labelme import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_dir")
    parser.add_argument("-o", "--out", default=None)
    args = parser.parse_args() 

    json_dir = args.json_dir

    jsonfiles = glob.glob(json_dir + '/' + "*.json")
    print(jsonfiles)

    if args.out is None:
      out_dir = osp.basename(json_dir)
    else:
      out_dir = args.out

    if not osp.exists(out_dir):
      os.mkdir(out_dir)

    labelpng_dir = out_dir + '/green_label'
    if not osp.exists(labelpng_dir):
      os.mkdir(labelpng_dir)

    for json_file in jsonfiles:
        data = json.load(open(json_file))
        imageData = data.get("imageData")

        if not imageData:
            imagePath = os.path.join(os.path.dirname(json_file), data["imagePath"])
            with open(imagePath, "rb") as f:
                imageData = f.read()
                imageData = base64.b64encode(imageData).decode("utf-8")
        img = utils.img_b64_to_arr(imageData)

        label_name_to_value = {"_background_": 0}
        for shape in sorted(data["shapes"], key=lambda x: x["label"]):
            label_name = shape["label"]
            if label_name in label_name_to_value:
                label_value = label_name_to_value[label_name]
            else:
                label_value = len(label_name_to_value)
                label_name_to_value[label_name] = label_value
        lbl, _ = utils.shapes_to_label(
            img.shape, data["shapes"], label_name_to_value
        )


        #PIL.Image.fromarray(img).save(osp.join(out_dir, "img.jpg"))
        folda_name = osp.split(json_file)
        file_name = osp.splitext(folda_name[-1])
        utils.lblsave(osp.join(labelpng_dir, file_name[0] + "_label.png"), lbl)

        logger.info("Saved to: {}".format(out_dir))


if __name__ == "__main__":
    main()
