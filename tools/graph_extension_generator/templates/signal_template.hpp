struct<SIGNAL_CLASS_NAME> : public<DS_PREFIX> INvDsSignal {
    /**
     * @brief Signal callback handler prototype
     */
    struct Handler {
        virtual<SIGNAL_RET_TYPE> on_<SIGNAL_FUNC_NAME>(<SIGNAL_PARAMS_WITH_TYPES>) = 0;
    };

    /**
     * @brief Set the signal handler
     */
    void set_handler(Handler *handler)
    {
        this->handler = handler;
        install_signal_handler();
    }

private:
    void install_signal_handler() override
    {
        if (element && element->get_element_ptr() && handler) {
            g_signal_connect(element->get_element_ptr(), "<SIGNAL_NAME>", G_CALLBACK(signal_cb),
                             this);
        }
    }

    static<SIGNAL_RET_TYPE> signal_cb(GstElement *element,
                                      <SIGNAL_PARAMS_WITH_TYPES_CB> gpointer data)
    {
        return reinterpret_cast << SIGNAL_CLASS_NAME > * >
               (data)->handler->on_<SIGNAL_FUNC_NAME>(<SIGNAL_PARAMS>);
    }

    Handler *handler;
};
