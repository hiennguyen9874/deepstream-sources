#include <vector>

#include "../debug_logger_raii.hpp"

// Sample function demonstrating debug logger usage
void processData(const std::vector<float> &data)
{
    // Start debug section
    {
        DEBUG_DUMP_SECTION();

        DEBUG_DUMP("Processing %zu elements", data.size());

        for (size_t i = 0; i < data.size(); i++) {
            if (data[i] > 0.5f) {
                DEBUG_DUMP("Element %zu: %f exceeds threshold", i, data[i]);
            }
        }

        DEBUG_DUMP("Processing complete");
    } // Debug logger automatically closes here
}

// Sample main function for testing
int main()
{
    std::vector<float> test_data = {0.1f, 0.6f, 0.3f, 0.8f, 0.2f};
    // expose DEBUG=1
    // unset DEBUG
    processData(test_data);
    return 0;
}
