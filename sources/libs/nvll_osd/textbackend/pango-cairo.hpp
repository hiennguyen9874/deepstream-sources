#ifndef BACKEND_PANGO_CAIRO_HPP
#define BACKEND_PANGO_CAIRO_HPP

#include "backend.hpp"

#ifdef ENABLE_TEXT_BACKEND_PANGO
std::shared_ptr<TextBackend> create_pango_cairo_backend();
#endif // ENABLE_TEXT_BACKEND_PANGO

#endif // BACKEND_PANGO_CAIRO_HPP
