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

import gi
import os
import pathlib
import uuid
import yaml

gi.require_version('GLib', '2.0')
gi.require_version('Gst', '1.0')

from gi.repository import Gst
from gi.repository import GObject

from graph_gstcomponent_generator import GraphGstComponentGenerator


class GraphGstExtensionGenerator:
    def __init__(self):
        basepath = str(pathlib.Path(__file__).parent.absolute())
        with open(basepath + "/templates/extension_template.cpp", 'r') as fh:
            self.__extensionTemplate = fh.read()
        with open(basepath + "/templates/extensionBUILD_template", 'r') as fh:
            self.__extensionBUILDTemplate = fh.read()

        self.__ext_def = {}

    def generateExtensionDefinitionYaml(self, autogenBasePath, pluginName):
        registry = Gst.Registry.get()
        dirName = autogenBasePath + '/' + pluginName
        if not os.path.exists(dirName):
            os.makedirs(dirName)
        extDefYamlPath = dirName + "/" + pluginName + "_extdef.yaml"
        self.__extDefYamlPath = extDefYamlPath

        if not os.path.exists(extDefYamlPath):
            plugin = registry.find_plugin(pluginName)
            if not plugin:
                print(f"Could not find plugin {plugin}")
                return

            extUuid = uuid.uuid3(uuid.uuid4(), dirName)
            self.__ext_def = {
                "name": pluginName,
                "uuid": extUuid.hex,
                "version": "0.0.1",
                "author": "AUTHOR",
                "license": "LICENSE",
                "url": "www.example.com",
                "description": plugin.get_description(),
                "labels" : ["nvidia", "gpu", "deepstream"],
            }
            self.__updateExtensionDefinitionYaml()

    def __updateExtensionDefinitionYaml(self):
        extensionDefinition = yaml.dump(self.__ext_def, sort_keys=False)
        with open(self.__extDefYamlPath, 'w') as fh:
            fh.write("---\n")
            fh.write(extensionDefinition)

    def generateExtension(self, autogenBasePath, pluginName, elementList):
        generator = GraphGstComponentGenerator()

        incls = []
        hdrs = []
        add_macros = []
        for elem in elementList:
            generator.generateComponentDefinitionYaml(autogenBasePath, elem)
            generator.generateComponent(autogenBasePath, elem)
            incls.append(f"#include \"{elem}.hpp\"")
            add_macros.append(f"GXF_EXT_FACTORY_ADD_{generator.getClassName()}();")
            for h in generator.getHeadersList():
                hdrs.append(f"\"{h}\"")

        dirName = autogenBasePath + '/' + pluginName
        if not os.path.exists(dirName):
            os.makedirs(dirName)

        extDefYamlPath = dirName + "/" + pluginName + "_extdef.yaml"
        with open(extDefYamlPath) as f:
            extDef = yaml.full_load(f)

        extCppPath = dirName + "/" + pluginName + "_ext.cpp"
        extCppContent = (self.__extensionTemplate
                         .replace("<HASH1>", "0x" + extDef["uuid"][:16])
                         .replace("<HASH2>", "0x" + extDef["uuid"][16:])
                         .replace("<EXTNAME>", extDef["name"])
                         .replace("<DESC>", extDef["description"])
                         .replace("<AUTHOR>", extDef["author"])
                         .replace("<VERSION>", extDef["version"])
                         .replace("<LICENSE>", extDef["license"])
                         .replace("<INCL_HDR>", "\n".join(incls))
                         .replace("<GXF_EXT_FACTORY_ADD_MACROS>", "\n\n".join(add_macros))
                        )
        with open(extCppPath, "w") as fh:
            fh.write(extCppContent)

        extuuid = extDef["uuid"]
        uuid_str = f"{extuuid[0:8]}-{extuuid[8:12]}-{extuuid[12:16]}-{extuuid[16:20]}-{extuuid[20:32]}"
        buildContent = (self.__extensionBUILDTemplate
                        .replace("<PLUGIN>", pluginName)
                        .replace("<VERSION>", extDef["version"])
                        .replace("<LICENSE>", extDef["license"])
                        .replace("<UUID>", uuid_str)
                        .replace("<URL>", extDef["url"])
                        .replace("<HDRS>", ",\n        ".join(hdrs))
                        .replace("<LABELS>", ",\n        ".join([f"\"{l}\"" for l in extDef["labels"]]))
                       )
        with open(dirName + "/BUILD", "w") as fh:
            fh.write(buildContent)
