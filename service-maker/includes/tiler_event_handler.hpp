/**
 * @file
 * <b>Help utililty to control Deepstream tiler during runtime </b>
 *
 * The tiler event handler support user control on tiler and osd during
 * runtime
 *
 */

#ifndef TILER_EVENT_HANDLER_HPP
#define TILER_EVEN_HANDLER_HPP

#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "element.hpp"

namespace deepstream {
class NvDsTilerEventHandler {
public:
    NvDsTilerEventHandler(Element *tiler, Element *osd, Element *renderer)
    {
        this->tiler = tiler;
        this->osd = osd;
        this->renderer = renderer;
    }

    ~NvDsTilerEventHandler() { stop(); }

    bool start();
    bool stop();

    bool handle_mouse_events_ = true;

    Element *tiler, *osd, *renderer;

    bool create_x_window();
    void destroy_x_window();
    void x_event_handler_thread_func();
    void kb_event_handler_thread_func();
    void *display = nullptr;
    uint64_t window = 0;
    bool x_event_thread_stop = false;
    std::thread x_event_thread;
    bool kb_event_thread_stop = false;
    std::thread kb_event_thread;
    bool kb_selecting = false;
    bool kb_row_selected = false;
    unsigned int selected_row, selected_col;
    int active_source_index = -1;
    std::string active_source_uri;

    void set_active_source(int sourceid);
    std::mutex mutex;
};
} // namespace deepstream

#endif