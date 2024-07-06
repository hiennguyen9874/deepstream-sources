#ifndef __NVDSCUSTOMLIB_FACTORY_HPP__
#define __NVDSCUSTOMLIB_FACTORY_HPP__

#include <dlfcn.h>
#include <errno.h>

#include <functional>
#include <iostream>

#include "nvdscustomlib_interface.hpp"

template <class T>
T *dlsym_ptr(void *handle, char const *name)
{
    return reinterpret_cast<T *>(dlsym(handle, name));
}

class DSCustomLibrary_Factory {
public:
    DSCustomLibrary_Factory() {}

    ~DSCustomLibrary_Factory()
    {
        if (m_libHandle) {
            dlclose(m_libHandle);
            m_libHandle = NULL;
            m_libName.clear();
        }
    }

    IDSCustomLibrary *CreateCustomAlgoCtx(std::string libName, GObject *object)
    {
        m_libName.assign(libName);

        m_libHandle = dlopen(m_libName.c_str(), RTLD_NOW);
        std::function<IDSCustomLibrary *(GObject *)> createAlgoCtx = nullptr;
        if (m_libHandle) {
            // std::cout << "Library Opened Successfully" << std::endl;

            createAlgoCtx =
                dlsym_ptr<IDSCustomLibrary *(GObject *)>(m_libHandle, "CreateCustomAlgoCtx");
            if (!createAlgoCtx) {
                throw std::runtime_error("createCustomAlgoCtx function not found in library");
            }
        } else {
            throw std::runtime_error(dlerror());
        }

        return createAlgoCtx ? createAlgoCtx(object) : nullptr;
    }

public:
    void *m_libHandle;
    std::string m_libName;
};

#endif
