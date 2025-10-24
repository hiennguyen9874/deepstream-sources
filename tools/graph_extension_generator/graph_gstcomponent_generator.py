# /usr/bin/env python3

################################################################################
# Copyright 2021-2022, NVIDIA CORPORATION
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

from ast import AsyncFunctionDef
import yaml
import uuid
import re
import pathlib

import os
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GLib', '2.0')
from gi.repository import GObject
from gi.repository import Gst
_DEBUG = True
_GXF_EXT_FACTORY_ADD_VERBOSE = True

# Map of GObject property type to GXF parameter type
GST_PROP_TYPE_MAP = {
    'gboolean': 'bool',
    'gchararray': 'std::string',
    'guint': 'uint64_t',
    'gint': 'int64_t',
    'guint64': 'uint64_t',
    'gint64': 'int64_t',
    'gulong': 'uint64_t',
    'glong': 'int64_t',
    'gfloat': 'double',
    'gdouble': 'double',
    'GstCaps': 'std::string',
    'GstStructure': 'std::string',
    'GstFraction': 'std::string',
}

# Map of GObject signal function prototype argument type to GXF parameter type
SIGNAL_PROP_TYPE_MAP = {
    'void': 'void',
    'gboolean': 'bool',
    'gchararray': 'gchar *',
    'gpointer': 'void *',
    'guint': 'unsigned int',
    'gint': 'int',
    'guint64': 'uint64_t',
    'gint64': 'int64_t',
    'guint32': 'uint32_t',
    'gint32': 'int32_t',
    'gulong': 'unsigned long',
    'glong': 'long',
    'gfloat': 'float',
    'gdouble': 'double',
    'GstBuffer': 'GstBuffer *',
    'GstPad': 'GstPad *'
}

# GStreamer element properties that should never to be added as GXF parameters
GST_PROP_SKIP_LIST = ['name', 'parent']

# Template code snipped to set GObject element Property set from GXF parameter
PROP_SET_TEMPLATE = 'auto p_<PROPNAME_US> = <PROPNAME_US>.try_get();\n\
    if (p_<PROPNAME_US> && p_<PROPNAME_US>.value() != <PROP_DEF_VALUE>) {\n\
      <GPROPTYPE> propvalue = (<GPROPTYPE>) p_<PROPNAME_US_VAL>;\n\
      g_object_set (element_, "<PROPNAME>", propvalue, NULL);\n\
    }\
'

PROP_SET_TEMPLATE_CAPS = 'auto p_<PROPNAME_US> = <PROPNAME_US>.try_get();\n\
    if (p_<PROPNAME_US>) {\n\
      GstCaps *propvalue = gst_caps_from_string(p_<PROPNAME_US>.value().c_str());\n\
      g_object_set (element_, "<PROPNAME>", propvalue, NULL);\n\
      gst_caps_unref(propvalue);\n\
    }\
'

PROP_SET_TEMPLATE_FRACTION = 'auto p_<PROPNAME_US> = <PROPNAME_US>.try_get();\n\
    if (p_<PROPNAME_US> && p_<PROPNAME_US>.value() != <PROP_DEF_VALUE>) {\n\
      GValue v = G_VALUE_INIT; \n\
      g_value_init (&v, GST_TYPE_FRACTION); \n\
      gst_value_deserialize (&v, p_<PROPNAME_US>.value().c_str()); \n\
      g_object_set_property (G_OBJECT (element_), "<PROPNAME>", &v);\n\
      g_value_unset (&v); \n\
    }\
'

PROP_SET_TEMPLATE_STRUCTURE = 'auto p_<PROPNAME_US> = <PROPNAME_US>.try_get();\n\
    if (p_<PROPNAME_US>) {\n\
      GstStructure *propvalue = \
          gst_structure_from_string(p_<PROPNAME_US>.value().c_str(), nullptr);\n\
      g_object_set (element_, "<PROPNAME>", propvalue, NULL);\n\
      gst_structure_free(propvalue);\n\
    }\
'

SET_PAD_DETAILS_TMPL = 'auto p_<PADNAME> = <PADNAME>.try_get();\n\
    if (p_<PADNAME>) {\n\
      p_<PADNAME>.value()->set_element(this);\n\
      p_<PADNAME>.value()->set_template_name("<PADTMPLNAME>");\n\
    }\
'

SIGNAL_SET_TEMPLATE = 'if (<SIGNAMEC>.try_get()) {\n\
      <SIGNAMEC>.try_get().value()->set_element(this);\n\
    }\
'

PROP_CONTROL_SET_TEMPLATE = '/**\n\
    * @brief Set method for \'<PROPNAME>\' (<PROPDESC>) \n\
    */\n\
    void set_<PROPNAME_US> (<PROPTYPE> <PROPNAME_US>) {\n\
        g_object_set(G_OBJECT(element_), "<PROPNAME>", <PROPNAME_US>, nullptr);\n\
    }'

PROP_CONTROL_GET_TEMPLATE = '/**\n\
    * @brief Get method for \'<PROPNAME>\' (<PROPDESC>) \n\
    */\n\
    void get_<PROPNAME_US> (<PROPTYPE> *<PROPNAME_US>) {\n\
        g_object_get(G_OBJECT(element_), "<PROPNAME>", <PROPNAME_US>, nullptr);\n\
    }'

# GXF component registration template code snippets for various types
GXF_EXT_FACTORY_ADD_TEMPLATE = 'GXF_EXT_FACTORY_ADD(<HASH_1>UL, <HASH_2>UL, \
    <NAMESPACE>::<SUB_NAMESPACE>::<CLASSNAME>, nvidia::deepstream::INvDsElement, "<DESC>");'
GXF_EXT_FACTORY_ADD_TEMPLATE_SIGNAL = 'GXF_EXT_FACTORY_ADD(<HASH_1>UL, <HASH_2>UL, \
    <NAMESPACE>::<SUB_NAMESPACE>::<CLASSNAME>, nvidia::deepstream::INvDsSignal, "<DESC>");'
GXF_EXT_FACTORY_ADD_TEMPLATE_ACTION = 'GXF_EXT_FACTORY_ADD(<HASH_1>UL, <HASH_2>UL, \
    <NAMESPACE>::<SUB_NAMESPACE>::<CLASSNAME>, nvidia::deepstream::INvDsAction, "<DESC>");'
GXF_EXT_FACTORY_ADD_TEMPLATE_PROPCONTROL = 'GXF_EXT_FACTORY_ADD(<HASH_1>UL, <HASH_2>UL, \
<NAMESPACE>::<SUB_NAMESPACE>::<CLASSNAME>PropertyController, \
nvidia::deepstream::INvDsPropertyController, \
"Helper component to control properties of <NAMESPACE>::<SUB_NAMESPACE>::<CLASSNAME>. \
This is a connector component. It must be linked to \'<CLASSNAME>\' for which \
properties are to be controlled and another component which will set/get properties \
via the helper component APIs.");'

# Verbose GXF component registration template code snippets for various types
GXF_EXT_FACTORY_ADD_VERBOSE_TEMPLATE = 'GXF_EXT_FACTORY_ADD_VERBOSE(<HASH_1>UL, <HASH_2>UL, \
    <NAMESPACE>::<SUB_NAMESPACE>::<CLASSNAME>, nvidia::deepstream::INvDsElement, "<DISPLAY_NAME>", "<BRIEF>", "<DESC>");'
GXF_EXT_FACTORY_ADD_VERBOSE_TEMPLATE_SIGNAL = 'GXF_EXT_FACTORY_ADD_VERBOSE(<HASH_1>UL, <HASH_2>UL, \
    <NAMESPACE>::<SUB_NAMESPACE>::<CLASSNAME>, nvidia::deepstream::INvDsSignal, "<DISPLAY_NAME>", "<BRIEF>", "<DESC>");'
GXF_EXT_FACTORY_ADD_VERBOSE_TEMPLATE_ACTION = 'GXF_EXT_FACTORY_ADD_VERBOSE(<HASH_1>UL, <HASH_2>UL, \
    <NAMESPACE>::<SUB_NAMESPACE>::<CLASSNAME>, nvidia::deepstream::INvDsAction, "<DISPLAY_NAME>", "<BRIEF>", "<DESC>");'

def elemNameToCompName(elementName):
    _USING_NEW_MWX = (
        "USE_NEW_NVSTREAMMUX" in os.environ and os.environ["USE_NEW_NVSTREAMMUX"] == "yes")
    if (elementName == "nvstreammux" or elementName == "nvstreamdemux") \
            and _USING_NEW_MWX:
        return elementName + "new"
    return elementName


def print_debug(*args):
    if _DEBUG:
        print(args)


Gst.init(None)


class GraphGstComponentGenerator:
    def __init__(self):
        basepath = str(pathlib.Path(__file__).parent.absolute())
        with open(basepath + "/templates/class_template.hpp", 'r') as fh:
            self.__classTemplate = fh.read()

        with open(basepath + "/templates/class_interface_template.hpp", 'r') as fh:
            self.__interfaceTemplate = fh.read()

        with open(basepath + "/templates/componentDefinitionYaml_template", 'r') as fh:
            self.__componentDefinitionYamlTemplate = fh.read()

        with open(basepath + "/templates/signal_template.hpp", 'r') as fh:
            self.__signalTemplate = fh.read()

        with open(basepath + "/templates/action_template.hpp", 'r') as fh:
            self.__actionTemplate = fh.read()

        with open(basepath + "/templates/runtime_control.hpp", 'r') as fh:
            self.__runtimeControlTemplate = fh.read()


    def __get_default_value(self, element, prop, proptype):
        defvalue = None
        try:
            if element:
                defvalue = element.get_property(prop.name)

            else:
                defvalue = prop.default_value
        except:
            defvalue = prop.default_value

        if prop.value_type.name == "gchararray" or str(prop.value_type.name) == "GstFraction":
            if defvalue:
                return f"std::string{{\"{defvalue}\"}}"
            else:
                return "std::string{\"\"}"

        elif prop.value_type.name == "gboolean":
            if str(defvalue) == "False":
                return "false"
            elif str(defvalue) == "True":
                return "true"
            else:
                return "gxf::Registrar::NoDefaultParameter()"

        elif str(prop.value_type.fundamental.name) == "GEnum":
            return f"{int(defvalue)}L"

        elif str(prop.value_type.name) == "GstCaps" or str(prop.value_type.name) == "GstStructure":
            return "gxf::Registrar::NoDefaultParameter()"

        type_suffix = ""

        if proptype == "guint64" or proptype == "guint":
            type_suffix = "UL"
        if proptype == "gint64" or proptype == "gint":
            type_suffix = "L"

        return f"{defvalue}{type_suffix}"

    def __generateSignal(self, elementName, signal, elemClassName):
        needCompDefUpdate = False

        isAction = signal.signal_flags & GObject.SIGNAL_ACTION
        signalName = signal.signal_name
        signalNameC = signalName.replace("-", "_")
        signalClassName = "NvDs" + \
            signalName.replace("-", " ").title().replace(" ", "")
        if isAction:
            signalClassName += "Action"
            if _GXF_EXT_FACTORY_ADD_VERBOSE:
                factory_add_code = GXF_EXT_FACTORY_ADD_VERBOSE_TEMPLATE_ACTION.replace(
                    "<CLASSNAME>", signalClassName)
            else:
                factory_add_code = GXF_EXT_FACTORY_ADD_TEMPLATE_ACTION.replace(
                    "<CLASSNAME>", signalClassName)
        else:
            signalClassName += "Signal"
            if _GXF_EXT_FACTORY_ADD_VERBOSE:
                factory_add_code = GXF_EXT_FACTORY_ADD_VERBOSE_TEMPLATE_SIGNAL.replace(
                    "<CLASSNAME>", signalClassName)
            else:
                factory_add_code = GXF_EXT_FACTORY_ADD_TEMPLATE_SIGNAL.replace(
                    "<CLASSNAME>", signalClassName)

        component_def = self.__comp_def

        if isAction:
            if "actions" not in component_def:
                component_def["actions"] = []
                needCompDefUpdate = True
            signal_defs = component_def["actions"]
        else:
            if "signals" not in component_def:
                component_def["signals"] = []
                needCompDefUpdate = True
            signal_defs = component_def["signals"]

        yamlSignalDef = None
        for sig in signal_defs:
            if sig["name"] == signalName:
                yamlSignalDef = sig

        if yamlSignalDef is None:
            yamlSignalDef = {}
            yamlSignalDef["name"] = signalName
            yamlSignalDef["uuid"] = str(uuid.uuid3(
                uuid.uuid4(), signalName)).replace("-", "")
            yamlSignalDef["description"] = signalName + \
                " placeholder description"
            yamlSignalDef["argument-names"] = []
            yamlSignalDef["ignore"] = False
            yamlSignalDef["brief"] = signalName + " placeholder brief"
            yamlSignalDef["display-name"] = signalName + " placeholder display-name"
            signal_defs.append(yamlSignalDef)
            needCompDefUpdate = True
        else:
            if "name" not in yamlSignalDef:
                yamlSignalDef["name"] = signalName
                needCompDefUpdate = True
            if "uuid" not in yamlSignalDef:
                yamlSignalDef["uuid"] = str(uuid.uuid3(
                    uuid.uuid4(), signalName)).replace("-", "")
                needCompDefUpdate = True
            if "description" not in yamlSignalDef:
                yamlSignalDef["description"] = signalName + \
                    " placeholder description"
                needCompDefUpdate = True
            if "argument-names" not in yamlSignalDef:
                yamlSignalDef["argument-names"] = []
                needCompDefUpdate = True
            if "ignore" not in yamlSignalDef:
                yamlSignalDef["ignore"] = False
                needCompDefUpdate = True
            if "brief" not in yamlSignalDef:
                yamlSignalDef["brief"] = signalName + " placeholder brief"
                needCompDefUpdate = True
            if "display-name" not in yamlSignalDef:
                yamlSignalDef["display-name"] = signalName + " placeholder display-name"
                needCompDefUpdate = True

        if yamlSignalDef["ignore"]:
            return (None, None, None, None, None)

        signalArgNames = yamlSignalDef["argument-names"]

        returnType = signal.return_type.name
        if signal.return_type.name in SIGNAL_PROP_TYPE_MAP:
            returnType = SIGNAL_PROP_TYPE_MAP[signal.return_type.name]

        params = []
        paramsWithTypes = []
        paramCount = 0
        for param in signal.param_types:
            if paramCount >= len(signalArgNames):
                signalArgNames.append("placeholderArgName" + str(paramCount))
                needCompDefUpdate = True
            paramName = signalArgNames[paramCount]
            params.append(paramName)

            paramType = param.name
            if param.name in SIGNAL_PROP_TYPE_MAP:
                paramType = SIGNAL_PROP_TYPE_MAP[param.name]

            paramsWithTypes.append(paramType + " " + paramName)
            paramCount += 1

        if signal.return_type.name == "void":
            actionReturnVariable = ""
            actionReturn = ""
            actionReturnVariablePass = ""
        else:
            actionReturnVariable = returnType + " ret;"
            actionReturn = "return ret;"
            actionReturnVariablePass = " &ret,"

        unique_id = yamlSignalDef["uuid"]
        hash_1 = "0x" + unique_id[:16]
        hash_2 = "0x" + unique_id[16:]
        factory_add_code = factory_add_code.replace("<HASH_1>", hash_1)
        factory_add_code = factory_add_code.replace("<HASH_2>", hash_2)

        if needCompDefUpdate:
            self.__updateComponentDefinitionYaml()

        if (isAction):
            desc = yamlSignalDef["description"] + \
                f". This is a connector component. This component must be linked to another component which triggers the action and '{elemClassName}' component which performs the action."
            signalDef = (self.__actionTemplate
                         .replace("<ACTION_CLASS_NAME>", signalClassName)
                         .replace("<ACTION_DESC>", desc)
                         .replace("<ACTION_RETURN_TYPE>", returnType)
                         .replace("<ACTION_PARAMS_WITH_TYPES>", ", ".join(paramsWithTypes))
                         .replace("<ACTION_RETURN_VARIABLE_DECL>", actionReturnVariable)
                         .replace("<ACTION_NAME>", signalName)
                         .replace("<ACTION_FUNC_NAME>", signalNameC)
                         .replace("<ACTION_PARAMS>", ", ".join(params))
                         .replace("<ACTION_RETURN_VARIABLE_PASS>", actionReturnVariablePass)
                         .replace("<ACTION_RETURN>", actionReturn)
                         )
        else:
            desc = yamlSignalDef["description"] + \
                f". This is a connector component. This component must be linked to '{elemClassName}' which emits the signal and another component which will handle the signal callback."
            signalParamsInCb = ", ".join(paramsWithTypes)
            if signalParamsInCb:
                signalParamsInCb += ", "
            signalDef = (self.__signalTemplate
                         .replace("<SIGNAL_CLASS_NAME>", signalClassName)
                         .replace("<SIGNAL_DESC>", desc)
                         .replace("<SIGNAL_RET_TYPE>", returnType)
                         .replace("<SIGNAL_PARAMS_WITH_TYPES>", ", ".join(paramsWithTypes))
                         .replace("<SIGNAL_PARAMS_WITH_TYPES_CB>",  signalParamsInCb)
                         .replace("<SIGNAL_NAME>", signalName)
                         .replace("<SIGNAL_FUNC_NAME>", signalNameC)
                         .replace("<SIGNAL_PARAMS>", ", ".join(params))
                         )
        factory_add_code = factory_add_code.replace("<DESC>", desc)
        if _GXF_EXT_FACTORY_ADD_VERBOSE:
            factory_add_code = factory_add_code.replace("<BRIEF>", yamlSignalDef["brief"])
            factory_add_code = factory_add_code.replace("<DISPLAY_NAME>", yamlSignalDef["display-name"])

        signal_decl_str = f"nvidia::gxf::Parameter<nvidia::gxf::Handle<{signalClassName}>> {signalNameC}_;"
        signalKeyName = signalName + ("-action" if isAction else "-signal")
        signal_register_str = ("result &= registrar->parameter(<PROPNAME>, \"<KEY>\", \"<HEADLINE>\", \"<SHORT_DESC>\", <DEFAULT_VALUE>, GXF_PARAMETER_FLAGS_OPTIONAL);"
                               .replace("<PROPNAME>", f"{signalNameC}_")
                               .replace("<KEY>", signalKeyName)
                               .replace("<HEADLINE>", signalKeyName.replace("-", " ").title())
                               .replace("<SHORT_DESC>", f"Handle to a {signalClassName} component")
                               .replace("<DEFAULT_VALUE>", "gxf::Registrar::NoDefaultParameter()")
                               )
        signal_set_str = SIGNAL_SET_TEMPLATE.replace(
            "<SIGNAMEC>", f"{signalNameC}_")
        return (signalDef, factory_add_code, signal_decl_str, signal_register_str, signal_set_str)

    def __generateClass(self, elementName):
        elem = Gst.ElementFactory.make(elementName, None)
        if elem is None:
            raise Exception("Element '" + elementName + "' not found")
        g_type = elem.g_type_instance.g_class.g_type
        className = g_type.name
        properties = elem.list_properties()
        pad_templates = elem.get_pad_template_list()
        declareParamsStringList = []
        registerParamsStringList = []
        setParamsStringList = []
        setPadDetailsStringList = []
        signalDefs = []

        component_def = self.__comp_def
        needDefUpdate = False

        if "class-name" not in component_def:
            component_def["class-name"] = className
            needDefUpdate = True
        className = component_def["class-name"]

        signal_ids = GObject.signal_list_ids(g_type)
        factory_add_macro_list = []
        for sig_id in signal_ids:
            signal = GObject.signal_query(sig_id)
            (signalDef, _factory_add_code, _signal_decl_str, _signal_register_str,
             _signal_set_str) = self.__generateSignal(elementName, signal, className)
            if signalDef == None:
                continue
            signalDefs.append(signalDef)
            factory_add_macro_list.append(_factory_add_code)

            declareParamsStringList.append(_signal_decl_str)
            registerParamsStringList.append(_signal_register_str)
            setParamsStringList.append(_signal_set_str)

        if "params" not in component_def:
            component_def["params"] = {}
            needDefUpdate = True
        if "list" not in component_def["params"]:
            component_def["params"]["list"] = []
            needDefUpdate = True
        if "default-values" not in component_def["params"]:
            component_def["params"]["default-values"] = {}
            needDefUpdate = True
        if "runtime-control-props-get" not in component_def["params"]:
            component_def["params"]["runtime-control-props-get"] = []
            needDefUpdate = True
        if "runtime-control-props-set" not in component_def["params"]:
            component_def["params"]["runtime-control-props-set"] = []
            needDefUpdate = True

        defParams = component_def["params"]["list"]
        defParamValues = component_def["params"]["default-values"]
        runtimeControlPropsGet = component_def["params"]["runtime-control-props-get"]
        runtimeControlPropsSet = component_def["params"]["runtime-control-props-set"]

        runtimeControlPropsCode = []

        if runtimeControlPropsGet or runtimeControlPropsSet:
            needRuntimeControl = True
        else:
            needRuntimeControl = False

        if needRuntimeControl:
            if "runtime-control-props-comp-details" not in component_def:
                component_def["runtime-control-props-comp-details"] = {}
            runtimeControlPropsComp = component_def["runtime-control-props-comp-details"]

        if "ignore" not in component_def["params"]:
            component_def["params"]["ignore"] = []
            needDefUpdate = True
        ignoreParamList = component_def["params"]["ignore"]

        if "mandatory" not in component_def["params"]:
            component_def["params"]["mandatory"] = []
            needDefUpdate = True
        mandatoryParamList = component_def["params"]["mandatory"]

        for prop in properties:
            if prop.name in GST_PROP_SKIP_LIST:
                continue
            if prop.name not in defParams:
                defParams.append(prop.name)
                needDefUpdate = True
            if prop.name in ignoreParamList:
                continue
            if not (prop.flags & GObject.ParamFlags.WRITABLE):
                ignoreParamList.append(prop.name)
                needDefUpdate = True
                continue
            propname_us = prop.name.replace("-", "_")

            enum_value_desc=""
            if prop.value_type.name in GST_PROP_TYPE_MAP:
                proptype = GST_PROP_TYPE_MAP[prop.value_type.name]
            elif prop.value_type.fundamental.name == 'GEnum':
                proptype = GST_PROP_TYPE_MAP['gint64']
                enum_value_desc = "\\n".join(["%3d: %s" % (pval, pname.value_nick) \
                    for pval, pname in prop.enum_class.__enum_values__.items()])
            else:
                proptype = "Unknown"

            param_decl_str = ("nvidia::gxf::Parameter<<PROPTYPE>> <PROPNAME>;"
                              .replace("<PROPNAME>", propname_us)
                              .replace("<PROPTYPE>", proptype)
                              )
            print_debug("propname_us = " + propname_us + " prop.name, value_type.n, fundamental.n= " +
                        prop.name + "," + prop.value_type.name + "," + prop.value_type.fundamental.name)
            param_desc = prop.blurb.replace('"', r'\"').replace(
                "\n", r' ').replace('\t', ' ')
            param_desc = re.sub(' +', ' ', param_desc)
            if enum_value_desc:
                param_desc += "\\nValid values:\\n" + enum_value_desc

            if prop.value_type.fundamental.name == 'GstFraction':
                param_desc += ". Format: <numerator>/<denominator>"


            if prop.name in mandatoryParamList:
                param_register_str = ("result &= registrar->parameter(<PROPNAME>, \"<KEY>\", \"<HEADLINE>\", \"<SHORT_DESC>\");"
                                    .replace("<PROPNAME>", propname_us)
                                    .replace("<KEY>", prop.name)
                                    .replace("<SHORT_DESC>", param_desc)
                                    .replace("<HEADLINE>", prop.nick)
                                    )
            else:
                param_register_str = ("result &= registrar->parameter(<PROPNAME>, \"<KEY>\", \"<HEADLINE>\", \"<SHORT_DESC>\", <DEFAULT_VALUE>, GXF_PARAMETER_FLAGS_OPTIONAL);"
                                    .replace("<PROPNAME>", propname_us)
                                    .replace("<KEY>", prop.name)
                                    .replace("<SHORT_DESC>", param_desc)
                                    .replace("<HEADLINE>", prop.nick)
                                    .replace("<DEFAULT_VALUE>", defParamValues[prop.name] if prop.name in defParamValues else self.__get_default_value(elem, prop, prop.value_type.name))
                                    )
            param_set_tmpl = PROP_SET_TEMPLATE
            if prop.value_type.name == 'GstCaps':
                param_set_tmpl = PROP_SET_TEMPLATE_CAPS
            if prop.value_type.name == 'GstFraction':
                param_set_tmpl = PROP_SET_TEMPLATE_FRACTION
            if prop.value_type.name == 'GstStructure':
                param_set_tmpl = PROP_SET_TEMPLATE_STRUCTURE

            param_set_str = (param_set_tmpl
                             .replace("<PROPNAME>", prop.name)
                             .replace("<PROPNAME_US>", propname_us)
                             .replace("<PROPNAME_US_VAL>", propname_us + ".value().c_str()" if prop.value_type.name == "gchararray" else propname_us + ".value()")
                             .replace("<GPROPTYPE>", "gint64" if prop.value_type.fundamental.name == 'GEnum' else prop.value_type.name)
                             .replace("<PROP_DEF_VALUE>", self.__get_default_value(elem, prop, prop.value_type.name))
                             )

            if proptype == 'Unknown' or prop.value_type.name == 'GstCaps':
                param_decl_str = '// ' + param_decl_str
                param_register_str = '// ' + param_register_str
                param_set_str = '/* ' + param_set_str + ' */'

            declareParamsStringList.append(param_decl_str)
            registerParamsStringList.append(param_register_str)
            setParamsStringList.append(param_set_str)

            rproptype = proptype
            if rproptype == "std::string":
                rproptype = "const char *"

            if prop.name in runtimeControlPropsSet:
                runtimeControlPropsCode.append(PROP_CONTROL_SET_TEMPLATE
                                               .replace("<PROPNAME_US>", propname_us)
                                               .replace("<PROPNAME>", prop.name)
                                               .replace("<PROPDESC>", param_desc)
                                               .replace("<PROPTYPE>", rproptype))

            if prop.name in runtimeControlPropsGet:
                runtimeControlPropsCode.append(PROP_CONTROL_GET_TEMPLATE
                                               .replace("<PROPNAME_US>", propname_us)
                                               .replace("<PROPNAME>", prop.name)
                                               .replace("<PROPDESC>", param_desc)
                                               .replace("<PROPTYPE>", rproptype))

        for ptmpl in pad_templates:
            pad_name = f"{ptmpl.name}_pad".replace("%", "")
            if (ptmpl.direction == Gst.PadDirection.SRC):
                if (ptmpl.presence == Gst.PadPresence.ALWAYS):
                    pad_type = "NvDsStaticOutput"
                elif (ptmpl.presence == Gst.PadPresence.SOMETIMES):
                    pad_type = "NvDsDynamicOutput"
                elif (ptmpl.presence == Gst.PadPresence.REQUEST):
                    pad_type = "NvDsOnRequestOutput"
                else:
                    pad_type = "Unknown"
            elif (ptmpl.direction == Gst.PadDirection.SINK):
                if (ptmpl.presence == Gst.PadPresence.ALWAYS):
                    pad_type = "NvDsStaticInput"
                elif (ptmpl.presence == Gst.PadPresence.SOMETIMES):
                    pad_type = "NvDsDynamicInput"
                elif (ptmpl.presence == Gst.PadPresence.REQUEST):
                    pad_type = "NvDsOnRequestInput"
                else:
                    pad_type = "Unknown"
            else:
                pad_type = "Unknown"

            hasVideo = False
            video_fmts = []
            hasAudio = False
            audio_fmts = []
            formats = ""

            if ptmpl.caps.is_any():
                formats = "ANY"
            else:
                formats = []

                def unique_append(l, v):
                    if v not in l:
                        l.append(v)

                for i in range(ptmpl.caps.get_size()):
                    cap = ptmpl.caps.get_structure(i)
                    cname = cap.get_name()
                    if "video/x-raw" in cname:
                        video_fmt_list = cap.get_list("format")
                        if video_fmt_list[1]:
                            for i in range(video_fmt_list[1].n_values):
                                unique_append(
                                    video_fmts, video_fmt_list[1].get_nth(i))
                        elif cap.get_string("format"):
                            unique_append(video_fmts, cap.get_string("format"))
                        hasVideo = True
                    elif "audio/x-raw" in cname:
                        audio_fmt_list = cap.get_list("format")
                        if audio_fmt_list[1]:
                            for i in range(audio_fmt_list[1].n_values):
                                unique_append(
                                    audio_fmts, audio_fmt_list[1].get_nth(i))
                        elif cap.get_string("format"):
                            unique_append(audio_fmts, cap.get_string("format"))
                        hasAudio = True
                    elif "video/x-" in cname:
                        unique_append(
                            video_fmts, cname.replace("video/x-", ""))
                    elif "audio/x-" in cname:
                        unique_append(
                            audio_fmts, cname.replace("audio/x-", ""))
                    elif "image/jpeg" in cname:
                        formats.append("jpeg")

                if elementName == "nvv4l2decoder" and ptmpl.name == "src":
                    video_fmts = ["NV12"]

                if video_fmts:
                    formats.append(f"video({', '.join(video_fmts)})")
                    hasVideo = True
                if audio_fmts:
                    formats.append(f"audio({', '.join(audio_fmts)})")
                    hasAudio = True
                formats = ";".join(formats)

            key = f"{ptmpl.name}".replace("_", "-")
            if hasVideo and not hasAudio:
                if not key.startswith("v"):
                    key = "video-" + key
                elif key.startswith("vsink") or key.startswith("vsrc"):
                    key = "video-" + key[1:]
            if hasAudio and not hasVideo:
                if not key.startswith("a"):
                    key = "audio-" + key
                elif key.startswith("asink") or key.startswith("asrc"):
                    key = "audio-" + key[1:]
            key = key.replace("src", "out").replace("sink", "in")

            param_decl_str = ("nvidia::gxf::Parameter<nvidia::gxf::Handle<<DS_PREFIX><PADTYPE>>> <PADNAME>;"
                              .replace("<PADNAME>", pad_name)
                              .replace("<PADTYPE>", pad_type)
                              )
            param_register_str = ("result &= registrar->parameter(<PADNAME>,\
                \"<KEY>\", \"<HEADLINE>\", \"<SHORT_DESC>\", <DEFAULT_VALUE>,\
                GXF_PARAMETER_FLAGS_OPTIONAL);"
                                  .replace("<PADNAME>", pad_name)
                                  .replace("<KEY>", key)
                                  .replace("<SHORT_DESC>", f"Handle to a nvidia::deepstream::{pad_type} component. Supported formats - {formats}")
                                  .replace("<HEADLINE>", key)
                                  .replace("<DEFAULT_VALUE>", "gxf::Registrar::NoDefaultParameter()")
                                  )
            pad_set_details_str = SET_PAD_DETAILS_TMPL.replace(
                "<PADNAME>", pad_name).replace("<PADTMPLNAME>", ptmpl.name)

            declareParamsStringList.append(param_decl_str)
            registerParamsStringList.append(param_register_str)
            setPadDetailsStringList.append(pad_set_details_str)

        if needRuntimeControl:

            signalDefs.append(self.__runtimeControlTemplate
                              .replace("<PROPMETHODS>", "\n\n".join(runtimeControlPropsCode))
                              .replace("<CLASSNAME>", className))

            if "uuid" not in runtimeControlPropsComp:
                runtimeControlPropsComp["uuid"] = str(
                    uuid.uuid3(uuid.uuid4(), className)).replace("-", "")
                needDefUpdate = True

            unique_id = runtimeControlPropsComp["uuid"]
            hash_1 = "0x" + unique_id[:16]
            hash_2 = "0x" + unique_id[16:]

            factory_add_macro_list.append(GXF_EXT_FACTORY_ADD_TEMPLATE_PROPCONTROL
                                          .replace("<CLASSNAME>", className)
                                          .replace("<HASH_1>", hash_1)
                                          .replace("<HASH_2>", hash_2))

            param_decl_str = f"nvidia::gxf::Parameter<nvidia::gxf::Handle<{className}PropertyController>> property_controller_;"
            param_register_str = f"result &= registrar->parameter(property_controller_, \"property-controller\", \"Property Controller\", \"Property Controller for {className} component\", gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);"
            param_set_str = '''if (property_controller_.try_get()) {
                property_controller_.try_get().value()->set_element(this);
            }'''

            declareParamsStringList.append(param_decl_str)
            registerParamsStringList.append(param_register_str)
            setParamsStringList.append(param_set_str)

        if _GXF_EXT_FACTORY_ADD_VERBOSE:
            factory_add_code = GXF_EXT_FACTORY_ADD_VERBOSE_TEMPLATE.replace(
                "<CLASSNAME>", className)
        else:
            factory_add_code = GXF_EXT_FACTORY_ADD_TEMPLATE.replace(
                "<CLASSNAME>", className)
        print_debug("get comonent definition for " + elementName)
        component_def = self.__comp_def
        for k, v in component_def.items():
            if k == "uuid":
                print_debug("uuid update to ", v)
                # replace HASH_1 and HASH_2
                unique_id = v
                hash_1 = "0x" + unique_id[:16]
                hash_2 = "0x" + unique_id[16:]
                factory_add_code = factory_add_code.replace("<HASH_1>", hash_1)
                factory_add_code = factory_add_code.replace("<HASH_2>", hash_2)
            if k == "description":
                # replace DESC
                print_debug("description update to ", v)
                factory_add_code = factory_add_code.replace("<DESC>", v)
            if k == "brief" and _GXF_EXT_FACTORY_ADD_VERBOSE:
                # replace BRIEF
                if v == "<COMPONENT_BRIEF>":
                    print_debug("User did not specify component brief")
                    v = className + " placeholder brief"
                print_debug("brief update to ", v)
                factory_add_code = factory_add_code.replace("<BRIEF>", v)
            if k == "display-name" and _GXF_EXT_FACTORY_ADD_VERBOSE:
                # replace DISPLAY_NAME
                if v == "<COMPONENT_DISPLAY_NAME>":
                    print_debug("User did not specify component display-name")
                    v = className + " placeholder display-name"
                print_debug("display-name update to ", v)
                factory_add_code = factory_add_code.replace("<DISPLAY_NAME>", v)

        if "extra-headers" not in component_def:
            component_def["extra-headers"] = []
            needDefUpdate = True
        extraHdrs = component_def["extra-headers"]

        if signalDefs and not extraHdrs:
            extraHdrs = [f"{elementName}_interfaces.hpp"]

        extraIncls = []
        for hdr in extraHdrs:
            extraIncls.append(f"#include \"{hdr}\"")

        if needDefUpdate:
            self.__updateComponentDefinitionYaml()

        factory_add_macro_list.append(factory_add_code)

        self.__className = className

        return (self.__classTemplate
                .replace("<ELEMENTNAME>", elementName)
                .replace("<CLASSNAME>", className)
                .replace("<EXTRA_HDRS>", "\n".join(extraIncls))
                .replace("<DECLPARAMS>", "\n  ".join(declareParamsStringList))
                .replace("<REGPARAMS>", "\n    ".join(registerParamsStringList))
                .replace("<PROPSET>", "\n\n    ".join(setParamsStringList))
                .replace("<SETPADDETAILS>", "\n\n    ".join(setPadDetailsStringList))
                .replace("<GXF_EXT_FACTORY_ADD_MACRO>", " \\\n\\\n".join(factory_add_macro_list)),
                "\n\n".join(signalDefs)
                )

    def __generateComponent(self, pluginDir, compDefinitionFile, elementName):
        print_debug("opening component definition: " + compDefinitionFile)
        with open(compDefinitionFile) as f:
            self.__comp_def = yaml.full_load(f)

        print_debug("Generating code for '{}'".format(elementName))

        if "namespace" not in self.__comp_def:
            self.__comp_def["namespace"] = "nvidia"
            self.__updateComponentDefinitionYaml()
        if "sub-namespace" not in self.__comp_def:
            self.__comp_def["sub-namespace"] = "deepstream"
            self.__updateComponentDefinitionYaml()

        nameSpace = self.__comp_def["namespace"]
        subNameSpace = self.__comp_def["sub-namespace"]

        (classCode, interfaceDefs) = self.__generateClass(elementName)
        classCode = (classCode.replace("<NAMESPACE>", nameSpace)
                     .replace("<SUB_NAMESPACE>", subNameSpace))
        if nameSpace == "nvidia" and subNameSpace == "deepstream":
            classCode = classCode.replace("<DS_PREFIX>", "")
        else:
            classCode = classCode.replace(
                "<DS_PREFIX>", "nvidia::deepstream::")

        compName = elemNameToCompName(elementName)

        compFile = f"{pluginDir}/{compName}.hpp"
        with open(compFile, "w")as fh:
            fh.write(classCode)
        os.system(f"which clang-format >/dev/null 2>&1 && clang-format -i --style=Google {compFile}")

        self.__hdrList = [f"{compName}.hpp"]

        if interfaceDefs:
            hdr_file_content = (self.__interfaceTemplate
                                .replace("<NAMESPACE>", nameSpace)
                                .replace("<SUB_NAMESPACE>", subNameSpace)
                                .replace("<SIGNALS>", interfaceDefs)
                                )
            if nameSpace == "nvidia" and subNameSpace == "deepstream":
                hdr_file_content = hdr_file_content.replace("<DS_PREFIX>", "")
            else:
                hdr_file_content = hdr_file_content.replace(
                    "<DS_PREFIX>", "nvidia::deepstream::")
            hdr_file_path = f"{pluginDir}/{compName}_interfaces.hpp"
            with open(hdr_file_path, "w") as fh:
                fh.write(hdr_file_content)
            os.system(f"which clang-format >/dev/null 2>&1 && clang-format -i --style=Google {hdr_file_path}")
            self.__hdrList.append(f"{compName}_interfaces.hpp")


    def getFeatureMetadata(self, registry, elementName, metaKey):
        feature = registry.lookup_feature(elementName)
        author = ""
        if isinstance(feature, Gst.ElementFactory):
            author += Gst.ElementFactory.get_metadata(feature, metaKey)
        if isinstance(feature, Gst.DeviceProviderFactory):
            author += Gst.DeviceProviderFactory.get_metadata(feature, metaKey)
        author = str(author.encode('ascii', 'ignore').decode('utf-8'))
        print_debug("author of " + feature.get_name() + " is " + author)
        return author

    def __updateComponentDefinitionYaml(self):
        componentDefinition = yaml.dump(self.__comp_def, sort_keys=False)
        with open(self.__compDefYamlPath, 'w') as fh:
            fh.write("\n---\n")
            fh.write(componentDefinition)

    def generateComponentDefinitionYaml(self, baseAutogenOutputDir, elementName):
        registry = Gst.Registry.get()

        feature = registry.lookup_feature(elementName)
        if feature is None:
            print(f"\033[31mCould not load element {elementName}\033[0m")
            return
        plugin = feature.get_plugin()
        elemPluginName = plugin.get_name()

        dirName = baseAutogenOutputDir + '/' + elemPluginName
        if not os.path.exists(dirName):
            os.makedirs(dirName)
        compName = elemNameToCompName(elementName)
        compDefYamlPath = dirName + "/" + compName + "_def.yaml"
        self.__compDefYamlPath = compDefYamlPath

        if not os.path.exists(compDefYamlPath):
            compUuid = uuid.uuid3(uuid.uuid4(), compName)
            componentDefinition = (self.__componentDefinitionYamlTemplate
                                   .replace("<COMPONENT_NAME>", compName)
                                   .replace("<COMPONENT_DESC>", self.getFeatureMetadata(registry, elementName, "description"))
                                   .replace("<COMPONENT_UUID>", compUuid.hex))
            with open(compDefYamlPath, 'w') as fh:
                fh.write(componentDefinition)


    def generateComponent(self, baseAutogenOutputDir, elementName):
        feature = Gst.Registry.get().lookup_feature(elementName)
        if feature is None:
            print(f"\033[31mCould not load element {elementName}\033[0m")
            return
        plugin = feature.get_plugin()
        elemPluginName = plugin.get_name()

        dirName = baseAutogenOutputDir + '/' + elemPluginName
        if not os.path.exists(dirName):
            os.makedirs(dirName)
        compName = elemNameToCompName(elementName)
        self.__generateComponent(
            dirName, '{}/{}_def.yaml'.format(dirName, compName), elementName)

    def getClassName(self):
        return self.__className

    def getHeadersList(self):
        return self.__hdrList
