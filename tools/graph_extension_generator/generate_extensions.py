# /usr/bin/env python3

################################################################################
# Copyright 2021, NVIDIA CORPORATION
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################

from graph_gstextension_generator import GraphGstExtensionGenerator

import pathlib
import shutil
import os
import sys
import gi

gi.require_version('GLib', '2.0')
gi.require_version('Gst', '1.0')
from gi.repository import Gst
from gi.repository import GObject

SCRIPT_DIR = str(pathlib.Path(__file__).parent.absolute())
TEMPLATE_DIR = SCRIPT_DIR + "/templates"

def groupElementsByPlugin(elementList):
    pluginGroups = {}
    registry = Gst.Registry.get()
    for elementName in elementList:
        feature = registry.lookup_feature(elementName)
        if feature is None:
            print(f"\033[31mCould not load element {elementName}\033[0m")
            continue
        plugin = feature.get_plugin()
        pluginName = plugin.get_name()
        if not pluginName in pluginGroups:
            pluginGroups[pluginName] = []
        pluginGroups[pluginName].append(elementName)
    return pluginGroups


if len(sys.argv) < 3:
    print("Usage: ./{} <element-file-list> <output-directory>".format(sys.argv[0].split('/')[-1]))
    sys.exit(-1)

elementListTemp = []
with open(sys.argv[1], 'r') as fh:
    elementListTemp = fh.read().splitlines()
    elementListTemp.sort()

Gst.init(None)

elementList = []
[elementList.append(x) for x in elementListTemp if x not in elementList and (
    not x.startswith("#"))]

pluginGroups = groupElementsByPlugin(elementList)

outputBasePath = sys.argv[2]
autogenBasePath = outputBasePath + '/extensions'


generator = GraphGstExtensionGenerator()

for pluginName, pluginElements in pluginGroups.items():
    generator.generateExtensionDefinitionYaml(autogenBasePath, pluginName)
    generator.generateExtension(autogenBasePath, pluginName, pluginElements)

shutil.copy2(TEMPLATE_DIR + "/WORKSPACE_template", outputBasePath + "/WORKSPACE")
shutil.copy2(TEMPLATE_DIR + "/bazelrc_template", outputBasePath + "/.bazelrc")
os.makedirs(outputBasePath + "/build/third_party/", exist_ok = True)
shutil.copy2(TEMPLATE_DIR + "/glib.BUILD", outputBasePath + "/build/third_party/glib.BUILD")
shutil.copy2(TEMPLATE_DIR + "/gstreamer.BUILD", outputBasePath + "/build/third_party/gstreamer.BUILD")
os.system(f"touch {outputBasePath}/build/BUILD")