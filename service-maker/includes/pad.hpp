/**
 * @file
 * <b>Pad definition </b>
 */

#ifndef NVIDIA_DEEPSTREAM_PAD
#define NVIDIA_DEEPSTREAM_PAD

#include "object.hpp"

namespace deepstream {

/**
 * @brief Pad is an abstraction of the I/O with an Element, @see Element
 *
 * Pad class derives from the base Object class, so it is reference based,
 * supports copying and moving.
 * A Pad instance must be either for input or for output
 *
 */
class Pad : public Object {
public:
    /** empty constructor */
    Pad();
    /** substantial constructor */
    Pad(bool is_input, const std::string &name = std::string());
    /** copy constructor */
    Pad(const Object &);
    /** move constructor */
    Pad(Object &&);
};
} // namespace deepstream

#endif