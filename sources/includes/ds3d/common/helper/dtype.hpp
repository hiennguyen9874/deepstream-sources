#ifndef DS3D_COMMON_HELPER_DTYPE_HPP
#define DS3D_COMMON_HELPER_DTYPE_HPP

namespace ds3d {

struct Int2 {
    int x, y;

    Int2() = default;
    Int2(int x, int y = 0) : x(x), y(y) {}
};

struct Int3 {
    int x, y, z;

    Int3() = default;
    Int3(int x, int y = 0, int z = 0) : x(x), y(y), z(z) {}
};

struct Int4 {
    int x, y, z, w;

    Int4() = default;
    Int4(int x, int y = 0, int z = 0, int w = 0) : x(x), y(y), z(z), w(w) {}
};

struct Float2 {
    float x, y;

    Float2() = default;
    Float2(float x, float y = 0) : x(x), y(y) {}
};

struct Float3 {
    float x, y, z;

    Float3() = default;
    Float3(float x, float y = 0, float z = 0) : x(x), y(y), z(z) {}
};

struct Float4 {
    float x, y, z, w;

    Float4() = default;
    Float4(float x, float y = 0, float z = 0, float w = 0) : x(x), y(y), z(z), w(w) {}
};

// It is only used to specify the type only, while hoping to avoid header file contamination.
typedef struct {
    unsigned short __x;
} half;

}; // namespace ds3d

#endif // DS3D_COMMON_HELPER_DTYPE_HPP