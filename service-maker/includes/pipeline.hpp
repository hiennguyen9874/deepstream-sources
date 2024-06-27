/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

/**
 * @file
 * <b>Pipeline definition </b>
 *
 * The Pipeline class serves as the foundation of Deepstream-based AI streaming applications.
 * Media streams flow through interconnected elements within a Pipeline instance, processed
 * via buffers and metadata.
 *
 */
#ifndef PIPELINE_HPP
#define PIPELINE_HPP

#include <memory>
#include <string>
#include <thread>

#include "buffer_probe.hpp"
#include "element.hpp"
#include "object.hpp"
#include "signal_handler.hpp"

namespace deepstream {

/**
 * @brief Pipeline class definition
 *
 * Pipelie class provide the high level interface for most of the functionalities.
 * It is recommended to use Pipeline API for creating applications unless there is
 * some specific features that are only supported in lower level APIs from Element,
 * Object, etc.
 *
 */
class Pipeline : public Object {
public:
    typedef enum { INVALID, EMPTY, READY, PAUSED, PLAYING } State;

    /**
     * @brief base class for pipeline message
     *
     * Pipeline message provides a mechanism for pipeline to
     * notify the application of something happening
     */
    class Message {
    public:
        virtual ~Message() {}
        uint32_t type() { return type_; }

    protected:
        uint32_t type_;

        Message(uint32_t type) : type_(type) {}
    };

    /**
     * @brief Pipeline message on source add/remove
     */
    class DynamicSourceMessage : public Message {
    public:
        DynamicSourceMessage(void *);

        inline bool isSourceAdded() const { return source_added_; }
        inline uint32_t getSourceId() const { return source_id_; }
        inline const std::string &getSensorId() const { return sensor_id_; }
        inline const std::string &getSensorName() const { return sensor_name_; }
        inline const std::string &getUri() const { return uri_; }

    protected:
        bool source_added_;
        uint32_t source_id_;
        std::string sensor_id_;
        std::string sensor_name_;
        std::string uri_;
    };

    /**
     * @brief Pipeline message on state transition
     */
    class StateTransitionMessage : public Message {
    public:
        StateTransitionMessage(void *);
        inline const std::string &getName() const { return name_; };
        inline void getState(State &old_state, State &new_state) const
        {
            old_state = old_state_;
            new_state = new_state_;
        }

    protected:
        State old_state_;
        State new_state_;
        std::string name_;
    };

    Pipeline(const char *name);

    /** @brief Constructor with name and a description file */
    Pipeline(const char *name, const std::string &config_file);

    /** @brief Destructor */
    virtual ~Pipeline();

    /** @brief  Template function for creating and adding element with properties */
    template <typename... Args>
    Pipeline &add(const std::string &element_type,
                  const std::string &element_name,
                  const Args &... args)
    {
        Element element = Element(element_type, element_name);
        if constexpr (sizeof...(args) > 0) {
            element.set(args...);
        }
        return this->add(element);
    }

    /** @brief Add a given element to the pipeline */
    Pipeline &add(Element element);

    /** @brief Find an element within the pipeline by name */
    Element *find(const ::std::string &name);

    /** @brief Operator for accessing elements in a pipeline */
    Element &operator[](const std::string &name)
    {
        Element *e = this->find(name);
        if (e == nullptr) {
            throw std::runtime_error(name + " Not found");
        }
        return *e;
    }

    /**
     * @brief Link two elements within the pipeline
     *
     * @param[in] route   a pair with source element name and target element name
     * @param[in] tips    extra pair with source pad name and target pad name
     */
    Pipeline &link(std::pair<std::string, std::string> route,
                   std::pair<std::string, std::string> tips);

    /** @brief Template function for linking elements in the simplest way */
    template <typename... Args>
    Pipeline &link(const std::string &arg1, const std::string arg2, const Args &... args)
    {
        (*this)[arg1].link((*this)[arg2]);
        if constexpr (sizeof...(args) > 0) {
            return this->link(arg2, args...);
        } else {
            return *this;
        }
    }

    /**
     * @brief Attach a custom object to an element within the pipeline
     *
     * Supported custom objects can be buffer probes, signal handlers
     * Once the object is attached, the element takes the ownership of it
     *
     * @param[in] element_name   name of the elment to which the object attaches
     * @param[in] object         pointer to a custom object
     * @param[in] tips           extra information. pad name for buffer probes, signal name for
     * signal handlers
     */
    Pipeline &attach(const std::string &elmenent_name,
                     CustomObject *object,
                     const std::string tip = "");

    /**
     * @brief Create and attach a custom object to an element within the pipeline
     *
     * Supported custom objects can be buffer probes, signal handlers
     * The custom object will be created through factory
     *
     * @param[in] element_name   name of the elment to which the object attaches
     * @param[in] plugin_name    name of the plugin where the custom object factory is defined
     * @param[in] object_name    name of the new custome object
     * @param[in] tips           extra information. pad name for buffer probes, signal name for
     * signal handlers
     */
    Pipeline &attach(const std::string &element_name,
                     const std::string &plugin_name,
                     const std::string &object_name,
                     const std::string tip = "");

    /** @brief Template function for creating and attaching custom object with properties */
    template <typename... Args>
    Pipeline &attach(const std::string &element_name,
                     const std::string &plugin_name,
                     const std::string &object_name,
                     const std::string tip,
                     const Args &... args)
    {
        attach(element_name, plugin_name, object_name, tip);
        if (sizeof...(args) > 0) {
            auto &element = (*this)[element_name];
            if (auto handler = element.getSignalHandler(object_name)) {
                handler->set(args...);
            } else if (auto probe = element.getProbe(object_name)) {
                probe->set(args...);
            }
        }
        return *this;
    }

    /** @brief install a callback to capture keyboard events */
    Pipeline &install(std::function<void(Pipeline &, int key)> keyboard_listener)
    {
        keyboard_listener_ = keyboard_listener;
        return *this;
    }

    /** @brief Start the pipeline */
    Pipeline &start();
    /** @brief Start the pipeline with a callback to capture the messages */
    Pipeline &start(std::function<void(Pipeline &, const Message &)> listener);
    /** @brief Wait until the pipeline ends */
    Pipeline &wait();

    /** @brief Stop the pipeline */
    Pipeline &stop();

    bool isRunning();

    /** @brief Pause the pipeline */
    Pipeline &pause();
    /** @brief Resume the pipeline */
    Pipeline &resume();

    void handleKey(int key);

protected:
    int run();

    // This is GMainLoop*
    void *loop_ = NULL;
    // gstreamer bus watch id
    uint bus_watch_id_;
    // gstreamer bus data
    void *bus_data_;
    std::function<void(Pipeline &, int key)> keyboard_listener_;

    // Thread
    std::thread thread_;
    // Hold all the element references here
    std::map<std::string, Element> elements_;
    // Hold the signal emitters for action control
    std::map<std::string, std::unique_ptr<SignalEmitter>> action_owners_;
};

} // namespace deepstream

#endif
