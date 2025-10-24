#pragma once

#include <string>

#include "nvdsbase/nvds_io.hpp"
<EXTRA_HDRS>

namespace <NAMESPACE>
{
    namespace<SUB_NAMESPACE>
    {
        class<CLASSNAME> : public<DS_PREFIX> INvDsElement {
        public:
            gxf_result_t initialize() override
            {
                GXF_LOG_INFO("initialize: %s %s\n", GST_ELEMENT_NAME, name());
                return GXF_SUCCESS;
            }

            gxf_result_t create_element() override
            {
                std::string ename = entity().name();
                ename += "/";
                ename += name();
                GXF_LOG_INFO("create_element: %s %s\n", GST_ELEMENT_NAME, name());
                element_ = gst_element_factory_make(GST_ELEMENT_NAME, ename.c_str());

                if (!element_) {
                    GXF_LOG_ERROR("Could not create GStreamer element '%s'", GST_ELEMENT_NAME);
                    return GXF_FAILURE;
                }

                <SETPADDETAILS> return GXF_SUCCESS;
            }

            gxf_result_t bin_add(GstElement *pipeline) override
            {
                GXF_LOG_INFO("bin_add: %s %s\n", GST_ELEMENT_NAME, name());
                if (!gst_bin_add(GST_BIN(pipeline), element_)) {
                    return GXF_FAILURE;
                }

                <PROPSET>

                    return GXF_SUCCESS;
            }

            GstElement *get_element_ptr() override { return element_; }

            gxf_result_t registerInterface(nvidia::gxf::Registrar *registrar) override
            {
                nvidia::gxf::Expected<void> result;
                <REGPARAMS> return nvidia::gxf::ToResultCode(result);
            }

            <DECLPARAMS>

                protected : GstElement *element_;
            const char *GST_ELEMENT_NAME = "<ELEMENTNAME>";
        };

#define GXF_EXT_FACTORY_ADD_                         \
    <CLASSNAME>() do { <GXF_EXT_FACTORY_ADD_MACRO> } \
    while (0)

    } // namespace >
} // namespace <NAMESPACE>
