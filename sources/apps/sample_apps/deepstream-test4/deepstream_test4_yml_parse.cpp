#include "deepstream_test4_yml_parse.h"

#include <yaml-cpp/yaml.h>

#include <iostream>
#include <string>

guint ds_test4_parse_meta_type(gchar *cfg_file_path, const char *group)
{
    std::string paramKey = "";

    auto docs = YAML::LoadAllFromFile(cfg_file_path);

    int total_docs = docs.size();
    guint val = 0;

    for (int i = 0; i < total_docs; i++) {
        if (docs[i][group]) {
            if (docs[i][group]["msg2p-newapi"]) {
                val = docs[i][group]["msg2p-newapi"].as<guint>();
                return val;
            }
        }
    }

    return 0;
}
