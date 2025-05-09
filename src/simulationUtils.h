#ifndef __SIMULATION_UTILS_H__
#define __SIMULATION_UTILS_H__

#include <string>
#include <cstdint>

using std::string;

#define PAGE_SIZE (4096)
typedef uint64_t Addr;

inline bool isPageAligned(Addr addr) {
  return addr % PAGE_SIZE == 0;
}

inline bool isPageSized(unsigned long size) {
  return size % PAGE_SIZE == 0;
}

namespace Simulator {

enum PageLocation{ NOT_PRESENT, IN_SSD, IN_CPU, IN_GPU, NOT_KNOWN, IN_GPU_LEAST };

enum MigrationDirection{ Others, G2S, S2G, C2G, G2C};

const std::string print_pagelocation_array [6] = {
    "Not_present", "In_ssd", "In_cpu", "In_gpu", "Not_Known", "In_gpu_least"
};

enum GPUPageTableEvcPolicy{ RANDOM, LRU, GUIDED };

enum MigPolicy{ DEEPUM, OURS };

/**
 * @brief
 */
class DataMovementHint {
  public:
    DataMovementHint(PageLocation from, PageLocation to, 
                     int issued_time, Tensor* tensor) :
        from(from), to(to), issued_time(issued_time), tensor(tensor) {
      Assert(to != NOT_KNOWN);
      barrier_end_time = -1;
    }
    bool operator<(const DataMovementHint& rhs) const {
      if (issued_time == rhs.issued_time) {
        return p_order < rhs.p_order;
      }
      else
        return issued_time < rhs.issued_time;
    }
    
    PageLocation from;
    PageLocation to;
    MigrationDirection direction = Others;
    string human_readable_hint;
    int issued_time;
    int barrier_end_time;
    long p_order;
    Tensor* tensor;
};

} // namespace Simulator

#endif
