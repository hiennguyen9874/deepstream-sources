struct<ACTION_CLASS_NAME> : public<DS_PREFIX> INvDsAction {
    /**
     * @brief <ACTION_DESC>
     */
    <ACTION_RETURN_TYPE><ACTION_FUNC_NAME>(<ACTION_PARAMS_WITH_TYPES>)
    {
        <ACTION_RETURN_VARIABLE_DECL> if (element && element->get_element_ptr())
            g_signal_emit_by_name(element->get_element_ptr(), "<ACTION_NAME>", <ACTION_PARAMS>,
                                  <ACTION_RETURN_VARIABLE_PASS> NULL);
<ACTION_RETURN>
    }
    void set_element(INvDsElement *element) override { this->element = element; }

private:
    INvDsElement *element;
};
