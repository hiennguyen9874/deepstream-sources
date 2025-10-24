/**
 * @file
 * <b>DataFeeder definition </b>
 *
 */
#ifndef NVIDIA_DEEPSTREAM_DATA_FEEDER
#define NVIDIA_DEEPSTREAM_DATA_FEEDER

#include <condition_variable>
#include <mutex>
#include <thread>

#include "buffer.hpp"
#include "signal_handler.hpp"

namespace deepstream {

/**
 * @brief A specific signal handler for feeding data
 *
 * Users must implement the IDataProvider interface to create a
 * data feeder.
 * Data feeder can only be attached to elements that supports
 * "need-data/enough-data" signal, typically an "appsrc".
 *
 **/
class DataFeeder : public SignalHandler {
public:
    /**
     * @brief required interface for a data feeder
     *
     * Invoked when the data is required by the element to which
     * the data feeder is attached.
     */
    class IDataProvider {
    public:
        /**
         * @brief Read a data block
         *
         * Implementation of this virtual method requires generating a
         * buffer based on the data request.
         *
         * @param[in]  feeder reference to the data feeder
         * @param[in]  size   number of bytes requested by the feeder
         * @param[out] eos    indication of "end of stream"
         * @return            a buffer with data
         */
        virtual Buffer read(DataFeeder &feeder, unsigned int size, bool &eos) = 0;
        virtual ~IDataProvider() {}
    };

    /**
     * @brief  Constructor
     *
     * Create a data feeder with a user implemented data provider interface
     *
     * @param[in] name        name of the instance
     * @param[in] handler     implementation of the IDataProvider interface
     */
    DataFeeder(const std::string &name, IDataProvider *provider);

    /**
     * @brief  Constructor
     *
     * Create a data feeder with a user implemented data provider interface
     *
     * @param[in] name        name of the instance
     * @param[in] factory     name of the factory to create the instance
     * @param[in] handler     implementation of the IDataProvider interface
     */
    DataFeeder(const std::string &name, const char *factory, IDataProvider *provider);

    /** @brief Destructor */
    virtual ~DataFeeder();

    /** @brief Start feeding, called by the pipeline */
    void startFeed(void *appsrc, unsigned int size);
    /** @brief Stop feeding, called by the pipeline */
    void stopFeed();
    /** @brief Complete the feeding, called by the feeding thread */
    bool doFeed(void *appsrc, unsigned int size);

protected:
    std::unique_ptr<IDataProvider> data_provider_;
    std::thread worker_;
    std::mutex mutex_;
    std::condition_variable cv_data_;
    unsigned int data_size_;
    bool data_ready_;
    void *appsrc_;
    bool eos_;
};
} // namespace deepstream

#endif