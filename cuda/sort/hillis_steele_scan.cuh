#ifndef HILLIS_STEELE_SCAN_H__
#define HILLIS_STEELE_SCAN_H__

template <class T>
void hillis_steele_scan(const T * const d_input, T * const d_newPos, T * const d_block_sums,
        const size_t N, const size_t binMask, const size_t gridSize, const size_t blockSize);

#endif