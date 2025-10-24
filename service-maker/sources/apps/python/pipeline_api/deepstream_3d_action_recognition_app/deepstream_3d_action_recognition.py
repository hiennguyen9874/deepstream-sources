#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

from pyservicemaker import Pipeline, Probe, BatchMetadataOperator, osd
from multiprocessing import Process
import os
import sys

PIPELINE_NAME = "deepstream-3d-action-recognition"

class AddLabel(BatchMetadataOperator):
  def __init__(self):
    super().__init__()

  def handle_metadata(self, batch_meta):
    for user_meta in batch_meta.preprocess_batch_items:
      preprocess_batch_meta = user_meta.as_preprocess_batch()
      if not preprocess_batch_meta:
        continue
      for roi_meta in preprocess_batch_meta.rois:
        for classifier_meta in roi_meta.classifier_items:
          num_labels = classifier_meta.n_labels
          for i in range(num_labels):
            label = classifier_meta.get_n_label(i)
            display_text = "Label: " + label

            text = osd.Text()
            text.display_text = display_text.encode('ascii')
            text.x_offset = int(roi_meta.roi.left)
            text.y_offset = max(int(roi_meta.roi.top - 10), 0)
            text.font.name = osd.FontFamily.Serif
            text.font.size = 12
            text.font.color = osd.Color(1.0, 1.0, 1.0, 1.0)
            text.set_bg_color = True
            text.bg_color = osd.Color(0.0, 0.0, 0.0, 1.0)

            display_meta = batch_meta.acquire_display_meta()
            display_meta.add_text(text)
            roi_meta.frame_meta.append(display_meta)



def main(file_path):
    file_ext = os.path.splitext(file_path)[1]
    if file_ext in [".yaml", ".yml"]:
        pipeline = Pipeline(PIPELINE_NAME, file_path)
        pipeline.attach("pgie", Probe("add_label_to_display", AddLabel())).start().wait()
    
    else:
        print("Invalid File Type: " + file_path + " Please provide a .yaml config file")



if __name__ == '__main__':
    # Check input arguments
    if len(sys.argv) != 2:
        sys.stderr.write("usage: %s <YAML config file>\n" % sys.argv[0])
        sys.exit(1)
    # pipeline.wait() in the main function is a blocking call due to which the KeyboardInterrupt may not be processed immediately.
    # we use Process from multiprocessing which runs the main function in a different process and processes KeyboardInterrupt immediately.
    process = Process(target=main, args=(sys.argv[1],))
    try:
        process.start()
        process.join()
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Terminating process...")
        process.terminate()