class<CLASSNAME> PropertyController : public<DS_PREFIX> INvDsPropertyController {
public:
    <PROPMETHODS>

        /**
         * @brief Set the element component whose properties will be controlled.
         * This method is called by <CLASSNAME> component.
         */
        void set_element(INvDsElement *element) override
    {
        this->element_ = element->get_element_ptr();
    }

private:
    GstElement *element_;
};