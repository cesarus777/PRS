#pragma once

#include <cstring>
#include <iomanip>
#include <unistd.h>

#include <array>
#include <bit>
#include <bitset>
#include <cassert>
#include <climits>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <ranges>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

using addr_t = uint32_t;

using byte_t = int8_t;
using half_t = int16_t;
using word_t = int32_t;
using ubyte_t = uint8_t;
using uhalf_t = uint16_t;
using uword_t = uint32_t;

constexpr size_t riscv_bytes_in_word = 4;
constexpr size_t riscv_bits_in_byte = 8;
constexpr size_t num_bytes_in_inst = 4;
constexpr size_t num_bits_in_inst = riscv_bits_in_byte * num_bytes_in_inst;
constexpr size_t pc_step = num_bytes_in_inst;
constexpr size_t xlen = 32;

constexpr size_t host_byte_bits = CHAR_BIT;

static_assert(num_bits_in_inst == sizeof(word_t) * host_byte_bits);

class memory_t final {
  using cell_t = unsigned long long;

  std::vector<cell_t> storage;

  static constexpr auto bytes_per_cell = sizeof(cell_t) / sizeof(byte_t);
  static constexpr auto halfs_per_cell = sizeof(cell_t) / sizeof(half_t);
  static constexpr auto words_per_cell = sizeof(cell_t) / sizeof(word_t);
  static constexpr auto storage_alignment = bytes_per_cell;

  static constexpr uword_t byte_mask = ~ubyte_t{0};
  static constexpr uword_t half_mask = ~uhalf_t{0};
  static constexpr uword_t word_mask = ~uword_t{0};

  static constexpr auto cell_byte_mask = static_cast<cell_t>(byte_mask);
  static constexpr auto cell_half_mask = static_cast<cell_t>(half_mask);
  static constexpr auto cell_word_mask = static_cast<cell_t>(word_mask);

  bool is_aligned_address(addr_t addr) { return addr % storage_alignment == 0; }

  void ensure_storage_capacity(addr_t addr) {
    // assert(is_aligned_address(addr));
    auto storage_idx = addr / bytes_per_cell;
    if (storage_idx < storage.size())
      return;

    storage.resize(storage_idx + 1, 0);
  }

  template <typename data_t> void write_impl(addr_t addr, data_t data) {
    static_assert(sizeof(cell_t) >= sizeof(data_t));
    ensure_storage_capacity(addr);
    auto storage_idx = addr / bytes_per_cell;
    auto &cell = *reinterpret_cast<data_t *>(
        reinterpret_cast<ubyte_t *>(&storage[storage_idx]) +
        addr % bytes_per_cell);
    cell &= ~cell_byte_mask;
    cell |= data;
    std::cerr << "[mem:0x" << std::hex << std::setw(8) << addr << "]"
              << " <== " << "0x" << std::setw(8) << data << std::dec
              << std::endl;
  }

public:
  memory_t() {}

  ubyte_t read_byte(addr_t addr) {
    ensure_storage_capacity(addr);
    ubyte_t data =
        *(reinterpret_cast<ubyte_t *>(&storage[addr / bytes_per_cell]) +
          addr % bytes_per_cell);
    return data;
  }

  uhalf_t read_half(addr_t addr) {
    ensure_storage_capacity(addr);
    uhalf_t data = *reinterpret_cast<uhalf_t *>(
        reinterpret_cast<ubyte_t *>(&storage[addr / bytes_per_cell]) +
        addr % bytes_per_cell);
    return data;
  }

  uword_t read_word(addr_t addr) {
    ensure_storage_capacity(addr);
    uword_t data = *reinterpret_cast<uword_t *>(
        reinterpret_cast<ubyte_t *>(&storage[addr / bytes_per_cell]) +
        addr % bytes_per_cell);
    return data;
  }

  void write_byte(addr_t addr, ubyte_t data) { write_impl(addr, data); }

  void write_half(addr_t addr, uhalf_t data) { write_impl(addr, data); }

  void write_word(addr_t addr, uword_t data) { write_impl(addr, data); }

  void load_section(const char *data, size_t size, addr_t addr) {
    assert((addr % bytes_per_cell) == 0);
    if (size % storage_alignment)
      size = (size / storage_alignment + 1) * storage_alignment;
    ensure_storage_capacity(addr + size);
    auto *data_end = data + size;
    while (data != data_end) {
      cell_t new_cell = 0;
      auto n = sizeof(cell_t) / sizeof(char);
      std::memcpy(&new_cell, data, n);
      storage[addr / bytes_per_cell] = new_cell;
      data += n;
      addr += n;
    }
  }

  void extend_upto(addr_t addr) {
    ensure_storage_capacity(addr);
  }
};

using reg_t = int32_t;
using ureg_t = uint32_t;

inline reg_t to_reg(ureg_t r) {
  static_assert(sizeof(reg_t) == sizeof(ureg_t));
  return std::bit_cast<reg_t>(r);
}

inline ureg_t to_ureg(reg_t r) {
  static_assert(sizeof(reg_t) == sizeof(ureg_t));
  return std::bit_cast<ureg_t>(r);
}

inline addr_t to_addr(reg_t r) {
  static_assert(sizeof(reg_t) == sizeof(addr_t));
  return std::bit_cast<ureg_t>(r);
}

enum class riscv_abi_reg_t {
  zero = 0,
  ra = 1,
  sp = 2,
  gp = 3,
  tp = 4,
  t0 = 5,
  t1 = 6,
  t2 = 7,
  s0 = 8,
  s1 = 9,
  a0 = 10,
  a1 = 11,
  a2 = 12,
  a3 = 13,
  a4 = 14,
  a5 = 15,
  a6 = 17,
  a7 = 17,
  s2 = 18,
  s3 = 19,
  s4 = 20,
  s5 = 21,
  s6 = 22,
  s7 = 23,
  s8 = 24,
  s9 = 25,
  s10 = 26,
  s11 = 27,
  t3 = 28,
  t4 = 29,
  t5 = 30,
  t6 = 31,
};

class reg_file_t final {
public:
  static constexpr size_t nregs = 32;

private:
  std::array<reg_t, nregs> regs = {0};
  ureg_t pc = 0x0;

public:
  reg_file_t() = default;

  reg_t read(size_t reg) const {
    assert(reg < nregs);
    return regs[reg];
  }

  reg_t read(riscv_abi_reg_t reg) const {
    return read(static_cast<size_t>(reg));
  }

  void write(size_t reg, reg_t data) {
    if (reg == 0)
      return;
    assert(reg < nregs);
    regs[reg] = data;
    std::cerr << 'x' << reg << " <== " << std::hex << std::setw(8) << data
              << std::dec << std::endl;
  }

  void write(riscv_abi_reg_t reg, reg_t data) {
    return write(static_cast<size_t>(reg), data);
  }

  ureg_t get_pc() const { return pc; }

  void set_pc(ureg_t new_pc) { pc = new_pc; }
};

using inst_t = uint32_t;

enum class opcode_t {
  // arithmetic
  ADD,
  SUB,
  OR,
  XOR,
  AND,
  SRL,
  SLL,
  SRA,
  // arithmetic with immediate
  ADDI,
  ORI,
  XORI,
  ANDI,
  SRLI,
  SLLI,
  SRAI,
  // jumps and calls
  JAL,
  JALR,
  BEQ,
  BNE,
  BLT,
  BGE,
  BLTU,
  BGEU,
  // loads/sotres
  LB,
  LH,
  LW,
  LBU,
  LHU,
  SB,
  SH,
  SW,
  // data flow
  SLT,
  SLTU,
  SLTI,
  SLTIU,
  // upper immediate
  LUI,
  AUIPC,
  // special ones
  FENCE,
  ECALL,
  EBREAK,
  // zicsr
  CSRRW,
  CSRRS,
  CSRRC,
  CSRRWI,
  CSRRSI,
  CSRRCI,
};

inline constexpr std::array opcodes = {
    // arithmetic
    opcode_t::ADD,
    opcode_t::SUB,
    opcode_t::OR,
    opcode_t::XOR,
    opcode_t::AND,
    opcode_t::SRL,
    opcode_t::SLL,
    opcode_t::SRA,
    // arithmetic with immediate
    opcode_t::ADDI,
    opcode_t::ORI,
    opcode_t::XORI,
    opcode_t::ANDI,
    opcode_t::SRLI,
    opcode_t::SLLI,
    opcode_t::SRAI,
    // jumps and calls
    opcode_t::JAL,
    opcode_t::JALR,
    opcode_t::BEQ,
    opcode_t::BNE,
    opcode_t::BLT,
    opcode_t::BGE,
    opcode_t::BLTU,
    opcode_t::BGEU,
    // loads/sotres
    opcode_t::LB,
    opcode_t::LH,
    opcode_t::LW,
    opcode_t::LBU,
    opcode_t::LHU,
    opcode_t::SB,
    opcode_t::SH,
    opcode_t::SW,
    // data flow
    opcode_t::SLT,
    opcode_t::SLTU,
    opcode_t::SLTI,
    opcode_t::SLTIU,
    // upper immediate
    opcode_t::LUI,
    opcode_t::AUIPC,
    // special ones
    opcode_t::FENCE,
    opcode_t::ECALL,
    opcode_t::EBREAK,
    // zicsr
    opcode_t::CSRRW,
    opcode_t::CSRRS,
    opcode_t::CSRRC,
    opcode_t::CSRRWI,
    opcode_t::CSRRSI,
    opcode_t::CSRRCI,
};

struct no_funct_t final {};

inline constexpr no_funct_t no_funct;

template <size_t bitwidth, size_t offset> class funct_t final {
  inst_t data;
  bool has_funct = true;

public:
  consteval funct_t(no_funct_t) noexcept : data(0), has_funct(false) {}

  consteval funct_t(inst_t funct) noexcept {
    *this = funct_t(std::bitset<bitwidth>(funct));
  }

  consteval funct_t(std::bitset<bitwidth> funct) noexcept
      : data(funct.to_ulong() << offset) {}

  consteval operator inst_t() const noexcept { return data; }

  consteval inst_t filter() noexcept {
    if (!has_funct)
      return 0;
    return std::bitset<bitwidth>().set().to_ullong() << offset;
  }
};

using funct3 = funct_t<3, 12>;
using funct7 = funct_t<7, 25>;
using opcode_encoding_t = funct_t<7, 0>;

inline constexpr auto no_funct3 = funct3(no_funct);
inline constexpr auto no_funct7 = funct7(no_funct);

inline consteval opcode_encoding_t opcode_bits(opcode_t opcode) {
  switch (opcode) {
  case opcode_t::ADD:
  case opcode_t::SUB:
  case opcode_t::OR:
  case opcode_t::XOR:
  case opcode_t::AND:
  case opcode_t::SRL:
  case opcode_t::SLL:
  case opcode_t::SRA:
  case opcode_t::SLT:
  case opcode_t::SLTU:
    return 0b0110011u;
  case opcode_t::ADDI:
  case opcode_t::ORI:
  case opcode_t::XORI:
  case opcode_t::ANDI:
  case opcode_t::SRLI:
  case opcode_t::SLLI:
  case opcode_t::SRAI:
  case opcode_t::SLTI:
  case opcode_t::SLTIU:
    return 0b0010011u;
  case opcode_t::JAL:
    return 0b1101111u;
  case opcode_t::JALR:
    return 0b1100111u;
  case opcode_t::LB:
  case opcode_t::LH:
  case opcode_t::LW:
  case opcode_t::LBU:
  case opcode_t::LHU:
    return 0b0000011u;
  case opcode_t::ECALL:
  case opcode_t::EBREAK:
    return 0b1110011u;
  case opcode_t::FENCE:
    return 0b0001111u;
  case opcode_t::BEQ:
  case opcode_t::BNE:
  case opcode_t::BLT:
  case opcode_t::BGE:
  case opcode_t::BLTU:
  case opcode_t::BGEU:
    return 0b1100011u;
  case opcode_t::SB:
  case opcode_t::SH:
  case opcode_t::SW:
    return 0b0100011u;
  case opcode_t::LUI:
    return 0b0110111u;
  case opcode_t::AUIPC:
    return 0b0010111u;
  case opcode_t::CSRRW:
  case opcode_t::CSRRS:
  case opcode_t::CSRRC:
  case opcode_t::CSRRWI:
  case opcode_t::CSRRSI:
  case opcode_t::CSRRCI:
    return 0b1110011u;
  default:
    assert(false);
  }
};

enum class encoding_t {
  RTYPE,
  ITYPE,
  STYPE,
  BTYPE,
  UTYPE,
  JTYPE,
  SYSTYPE,
  CSRTYPE,
  CSRITYPE,
};

inline constexpr encoding_t encoding4opcode(opcode_t opcode) {
  switch (opcode) {
  case opcode_t::ADD:
  case opcode_t::SUB:
  case opcode_t::OR:
  case opcode_t::XOR:
  case opcode_t::AND:
  case opcode_t::SRL:
  case opcode_t::SLL:
  case opcode_t::SRA:
  case opcode_t::SLT:
  case opcode_t::SLTU:
    return encoding_t::RTYPE;
  case opcode_t::ADDI:
  case opcode_t::ORI:
  case opcode_t::XORI:
  case opcode_t::ANDI:
  case opcode_t::SRLI:
  case opcode_t::SLLI:
  case opcode_t::SRAI:
  case opcode_t::JALR:
  case opcode_t::LB:
  case opcode_t::LH:
  case opcode_t::LW:
  case opcode_t::LBU:
  case opcode_t::LHU:
  case opcode_t::SLTI:
  case opcode_t::SLTIU:
  case opcode_t::FENCE:
    return encoding_t::ITYPE;
  case opcode_t::JAL:
    return encoding_t::JTYPE;
  case opcode_t::BEQ:
  case opcode_t::BNE:
  case opcode_t::BLT:
  case opcode_t::BGE:
  case opcode_t::BLTU:
  case opcode_t::BGEU:
    return encoding_t::BTYPE;
  case opcode_t::SB:
  case opcode_t::SH:
  case opcode_t::SW:
    return encoding_t::STYPE;
  case opcode_t::LUI:
  case opcode_t::AUIPC:
    return encoding_t::UTYPE;
  case opcode_t::ECALL:
  case opcode_t::EBREAK:
    return encoding_t::SYSTYPE;
  case opcode_t::CSRRW:
  case opcode_t::CSRRS:
  case opcode_t::CSRRC:
    return encoding_t::CSRTYPE;
  case opcode_t::CSRRWI:
  case opcode_t::CSRRSI:
  case opcode_t::CSRRCI:
    return encoding_t::CSRITYPE;
  default:
    assert(false);
  }
}

template <class... ts_t> struct overloaded_t final : ts_t... {
  using ts_t::operator()...;
};

template <class... ts_t> auto overloaded(ts_t &&...ts) {
  return overloaded_t{std::forward<ts_t>(ts)...};
}

class reg_operand_info_t {
  bool dst;

public:
  reg_operand_info_t(bool is_dst) noexcept : dst(is_dst) {}

  auto is_dst() const noexcept { return dst; }
};

class imm_operand_info_t {
public:
  imm_operand_info_t() noexcept {}
};

class csr_operand_info_t {
public:
  csr_operand_info_t() noexcept {}
};

class operand_info_t {
  std::variant<reg_operand_info_t, imm_operand_info_t, csr_operand_info_t> info;

public:
  operand_info_t(reg_operand_info_t reg_info) : info(reg_info) {}
  operand_info_t(imm_operand_info_t imm_info) : info(imm_info) {}
  operand_info_t(csr_operand_info_t csr_info) : info(csr_info) {}

  auto &get() const & { return info; }

  template <typename self_t, typename... fs_t>
  auto visit(this self_t &&self, fs_t &&...funcs) {
    return std::visit(overloaded(std::forward<fs_t>(funcs)...),
                      std::forward<self_t>(self).info);
  }
};

class reg_operand_t final : private reg_operand_info_t {
  size_t reg;

public:
  reg_operand_t(reg_operand_info_t op_info, size_t reg_num)
      : reg_operand_info_t(op_info), reg(reg_num) {
    assert(reg_num < reg_file_t::nregs);
  }

  auto get_reg() const { return reg; }

  using reg_operand_info_t::is_dst;
};

class imm_operand_t final : private imm_operand_info_t {
  reg_t imm_data;

public:
  imm_operand_t(imm_operand_info_t info, reg_t imm)
      : imm_operand_info_t(info), imm_data(imm) {}

  auto imm() const noexcept { return imm_data; }
};

enum class csr_t {};

class csr_operand_t final : private csr_operand_info_t {
  csr_t csr_data;

public:
  csr_operand_t(csr_operand_info_t info, csr_t csr)
      : csr_operand_info_t(info), csr_data(csr) {}

  auto csr() const noexcept { return csr_data; }
};

class operand_t final {
  using op_var_t = std::variant<reg_operand_t, imm_operand_t, csr_operand_t>;
  op_var_t op;

public:
  operand_t(reg_operand_info_t op_info, size_t reg_num)
      : op(reg_operand_t(op_info, reg_num)) {}

  operand_t(imm_operand_info_t op_info, reg_t imm)
      : op(imm_operand_t(op_info, imm)) {}

  operand_t(csr_operand_info_t op_info, csr_t csr)
      : op(csr_operand_t(op_info, csr)) {}

  template <typename self_t, typename... fs_t>
  auto visit(this self_t &&self, fs_t &&...funcs) {
    return std::visit(overloaded(std::forward<fs_t>(funcs)...),
                      std::forward<self_t>(self).op);
  }
};

class reg_operand_matcher_t {
  inst_t filter_mask;
  size_t offset;

public:
  reg_operand_matcher_t(inst_t filter, size_t offset) noexcept
      : filter_mask(filter), offset(offset) {}

  size_t extract_reg(inst_t inst) const noexcept {
    return (inst & filter_mask) >> offset;
  }
};

class imm_operand_matcher_t {
public:
  enum class bit_map_t {
    TO_0 = 0,
    TO_1 = 1,
    TO_2 = 2,
    TO_3 = 3,
    TO_4 = 4,
    TO_5 = 5,
    TO_6 = 6,
    TO_7 = 7,
    TO_8 = 8,
    TO_9 = 9,
    TO_10 = 10,
    TO_11 = 11,
    TO_12 = 12,
    TO_13 = 13,
    TO_14 = 14,
    TO_15 = 15,
    TO_16 = 16,
    TO_17 = 17,
    TO_18 = 18,
    TO_19 = 19,
    TO_20 = 20,
    TO_21 = 21,
    TO_22 = 22,
    TO_23 = 23,
    TO_24 = 24,
    TO_25 = 25,
    TO_26 = 26,
    TO_27 = 27,
    TO_28 = 28,
    TO_29 = 29,
    TO_30 = 30,
    TO_31 = 31,
    ALWAYS_0,
  };

private:
  std::array<bit_map_t, num_bits_in_inst> bit_mapping;

  static bool get_bit(inst_t val, bit_map_t idx) {
    constexpr size_t bitwidth = num_bits_in_inst;
    if (idx == bit_map_t::ALWAYS_0)
      return 0;
    return std::bitset<bitwidth>(val)[static_cast<size_t>(idx)];
  }

  static reg_t set_bit(reg_t val, size_t idx, bool bit) {
    constexpr size_t bitwidth = xlen;
    std::bitset<bitwidth> bitval(std::bit_cast<ureg_t>(val));
    bitval[static_cast<size_t>(idx)] = bit;
    ureg_t uval = bitval.to_ulong();
    return std::bit_cast<reg_t>(uval);
  }

public:
  imm_operand_matcher_t(std::array<bit_map_t, num_bits_in_inst> mapping)
      : bit_mapping{std::move(mapping)} {}

  reg_t extract_imm(inst_t inst) const {
    reg_t imm = 0;
    for (auto [tgt_idx, src_idx] : bit_mapping | std::views::enumerate)
      imm = set_bit(imm, tgt_idx, get_bit(inst, src_idx));
    return imm;
  }

  static const std::unordered_map<encoding_t, imm_operand_matcher_t> per_enc;
};

inline const std::unordered_map<encoding_t, imm_operand_matcher_t>
    imm_operand_matcher_t::per_enc = {
        {encoding_t::ITYPE,
         std::array{bit_map_t::TO_20, bit_map_t::TO_21, bit_map_t::TO_22,
                    bit_map_t::TO_23, bit_map_t::TO_24, bit_map_t::TO_25,
                    bit_map_t::TO_26, bit_map_t::TO_27, bit_map_t::TO_28,
                    bit_map_t::TO_29, bit_map_t::TO_30, bit_map_t::TO_31,
                    bit_map_t::TO_31, bit_map_t::TO_31, bit_map_t::TO_31,
                    bit_map_t::TO_31, bit_map_t::TO_31, bit_map_t::TO_31,
                    bit_map_t::TO_31, bit_map_t::TO_31, bit_map_t::TO_31,
                    bit_map_t::TO_31, bit_map_t::TO_31, bit_map_t::TO_31,
                    bit_map_t::TO_31, bit_map_t::TO_31, bit_map_t::TO_31,
                    bit_map_t::TO_31, bit_map_t::TO_31, bit_map_t::TO_31,
                    bit_map_t::TO_31, bit_map_t::TO_31}},
        {encoding_t::STYPE,
         std::array{bit_map_t::TO_7,  bit_map_t::TO_8,  bit_map_t::TO_9,
                    bit_map_t::TO_10, bit_map_t::TO_11, bit_map_t::TO_25,
                    bit_map_t::TO_26, bit_map_t::TO_27, bit_map_t::TO_28,
                    bit_map_t::TO_29, bit_map_t::TO_30, bit_map_t::TO_31,
                    bit_map_t::TO_31, bit_map_t::TO_31, bit_map_t::TO_31,
                    bit_map_t::TO_31, bit_map_t::TO_31, bit_map_t::TO_31,
                    bit_map_t::TO_31, bit_map_t::TO_31, bit_map_t::TO_31,
                    bit_map_t::TO_31, bit_map_t::TO_31, bit_map_t::TO_31,
                    bit_map_t::TO_31, bit_map_t::TO_31, bit_map_t::TO_31,
                    bit_map_t::TO_31, bit_map_t::TO_31, bit_map_t::TO_31,
                    bit_map_t::TO_31, bit_map_t::TO_31}},
        {encoding_t::BTYPE,
         std::array{bit_map_t::ALWAYS_0, bit_map_t::TO_8,  bit_map_t::TO_9,
                    bit_map_t::TO_10,    bit_map_t::TO_11, bit_map_t::TO_25,
                    bit_map_t::TO_26,    bit_map_t::TO_27, bit_map_t::TO_28,
                    bit_map_t::TO_29,    bit_map_t::TO_30, bit_map_t::TO_7,
                    bit_map_t::TO_31,    bit_map_t::TO_31, bit_map_t::TO_31,
                    bit_map_t::TO_31,    bit_map_t::TO_31, bit_map_t::TO_31,
                    bit_map_t::TO_31,    bit_map_t::TO_31, bit_map_t::TO_31,
                    bit_map_t::TO_31,    bit_map_t::TO_31, bit_map_t::TO_31,
                    bit_map_t::TO_31,    bit_map_t::TO_31, bit_map_t::TO_31,
                    bit_map_t::TO_31,    bit_map_t::TO_31, bit_map_t::TO_31,
                    bit_map_t::TO_31,    bit_map_t::TO_31}},
        {encoding_t::UTYPE,
         std::array{
             bit_map_t::ALWAYS_0, bit_map_t::ALWAYS_0, bit_map_t::ALWAYS_0,
             bit_map_t::ALWAYS_0, bit_map_t::ALWAYS_0, bit_map_t::ALWAYS_0,
             bit_map_t::ALWAYS_0, bit_map_t::ALWAYS_0, bit_map_t::ALWAYS_0,
             bit_map_t::ALWAYS_0, bit_map_t::ALWAYS_0, bit_map_t::ALWAYS_0,
             bit_map_t::TO_12,    bit_map_t::TO_13,    bit_map_t::TO_14,
             bit_map_t::TO_15,    bit_map_t::TO_16,    bit_map_t::TO_17,
             bit_map_t::TO_18,    bit_map_t::TO_19,    bit_map_t::TO_20,
             bit_map_t::TO_21,    bit_map_t::TO_22,    bit_map_t::TO_23,
             bit_map_t::TO_24,    bit_map_t::TO_25,    bit_map_t::TO_26,
             bit_map_t::TO_27,    bit_map_t::TO_28,    bit_map_t::TO_29,
             bit_map_t::TO_30,    bit_map_t::TO_31}},
        {encoding_t::JTYPE,
         std::array{bit_map_t::ALWAYS_0, bit_map_t::TO_21, bit_map_t::TO_22,
                    bit_map_t::TO_23,    bit_map_t::TO_24, bit_map_t::TO_25,
                    bit_map_t::TO_26,    bit_map_t::TO_27, bit_map_t::TO_28,
                    bit_map_t::TO_29,    bit_map_t::TO_30, bit_map_t::TO_20,
                    bit_map_t::TO_12,    bit_map_t::TO_13, bit_map_t::TO_14,
                    bit_map_t::TO_15,    bit_map_t::TO_16, bit_map_t::TO_17,
                    bit_map_t::TO_18,    bit_map_t::TO_19, bit_map_t::TO_31,
                    bit_map_t::TO_31,    bit_map_t::TO_31, bit_map_t::TO_31,
                    bit_map_t::TO_31,    bit_map_t::TO_31, bit_map_t::TO_31,
                    bit_map_t::TO_31,    bit_map_t::TO_31, bit_map_t::TO_31,
                    bit_map_t::TO_31,    bit_map_t::TO_31}},
        {encoding_t::CSRITYPE,
         std::array{
             bit_map_t::TO_0,     bit_map_t::TO_1,     bit_map_t::TO_2,
             bit_map_t::TO_3,     bit_map_t::TO_4,     bit_map_t::ALWAYS_0,
             bit_map_t::ALWAYS_0, bit_map_t::ALWAYS_0, bit_map_t::ALWAYS_0,
             bit_map_t::ALWAYS_0, bit_map_t::ALWAYS_0, bit_map_t::ALWAYS_0,
             bit_map_t::ALWAYS_0, bit_map_t::ALWAYS_0, bit_map_t::ALWAYS_0,
             bit_map_t::ALWAYS_0, bit_map_t::ALWAYS_0, bit_map_t::ALWAYS_0,
             bit_map_t::ALWAYS_0, bit_map_t::ALWAYS_0, bit_map_t::ALWAYS_0,
             bit_map_t::ALWAYS_0, bit_map_t::ALWAYS_0, bit_map_t::ALWAYS_0,
             bit_map_t::ALWAYS_0, bit_map_t::ALWAYS_0, bit_map_t::ALWAYS_0,
             bit_map_t::ALWAYS_0, bit_map_t::ALWAYS_0, bit_map_t::ALWAYS_0,
             bit_map_t::ALWAYS_0, bit_map_t::ALWAYS_0}},
};

class csr_operand_matcher_t {
public:
  consteval csr_operand_matcher_t() noexcept {}

  csr_t extract_csr(inst_t inst) const noexcept {
    constexpr size_t bitwidth = 12;
    constexpr size_t offset = 20;
    constexpr inst_t filter_mask = std::bitset<bitwidth>().set().to_ullong()
                                   << 20;

    return static_cast<csr_t>((inst & filter_mask) >> offset);
  }
};

class operand_matcher_t {
  std::variant<reg_operand_matcher_t, imm_operand_matcher_t,
               csr_operand_matcher_t>
      matcher;

public:
  operand_matcher_t(reg_operand_matcher_t m) : matcher(std::move(m)) {}
  operand_matcher_t(imm_operand_matcher_t m) : matcher(std::move(m)) {}
  operand_matcher_t(csr_operand_matcher_t m) : matcher(std::move(m)) {}

  auto &get() const & { return matcher; }

  template <typename self_t, typename... fs_t>
  auto visit(this self_t &&self, fs_t &&...funcs) {
    return std::visit(overloaded(std::forward<fs_t>(funcs)...),
                      std::forward<self_t>(self).matcher);
  }
};

class operand_description_t {
  operand_info_t op_info;
  operand_matcher_t op_matcher;

public:
  operand_description_t(operand_info_t info, operand_matcher_t matcher) noexcept
      : op_info(info), op_matcher(std::move(matcher)) {}

  template <typename self_t, typename... fs_t>
  auto visit(this self_t &&self, fs_t &&...funcs) {
    auto op_info = std::forward<self_t>(self).op_info.get();
    auto op_matcher = std::forward<self_t>(self).op_matcher.get();
    return std::visit(overloaded(std::forward<fs_t>(funcs)...), op_info,
                      op_matcher);
  }

  static const std::unordered_map<encoding_t,
                                  std::vector<operand_description_t>>
      per_enc;
};

inline const std::unordered_map<encoding_t, std::vector<operand_description_t>>
    operand_description_t::per_enc = {
        {encoding_t::RTYPE,
         {
             {reg_operand_info_t(true), reg_operand_matcher_t(0x00000f80, 7)},
             {reg_operand_info_t(false), reg_operand_matcher_t(0x000f8000, 15)},
             {reg_operand_info_t(false), reg_operand_matcher_t(0x01f00000, 20)},
         }},
        {encoding_t::ITYPE,
         {
             {reg_operand_info_t(true), reg_operand_matcher_t(0x00000f80, 7)},
             {reg_operand_info_t(false), reg_operand_matcher_t(0x000f8000, 15)},
             {imm_operand_info_t(),
              imm_operand_matcher_t::per_enc.at(encoding_t::ITYPE)},
         }},
        {encoding_t::STYPE,
         {
             {reg_operand_info_t(false), reg_operand_matcher_t(0x000f8000, 15)},
             {reg_operand_info_t(false), reg_operand_matcher_t(0x01f00000, 20)},
             {imm_operand_info_t(),
              imm_operand_matcher_t::per_enc.at(encoding_t::STYPE)},
         }},
        {encoding_t::BTYPE,
         {
             {reg_operand_info_t(false), reg_operand_matcher_t(0x000f8000, 15)},
             {reg_operand_info_t(false), reg_operand_matcher_t(0x01f00000, 20)},
             {imm_operand_info_t(),
              imm_operand_matcher_t::per_enc.at(encoding_t::BTYPE)},
         }},
        {encoding_t::UTYPE,
         {
             {reg_operand_info_t(true), reg_operand_matcher_t(0x00000f80, 7)},
             {imm_operand_info_t(),
              imm_operand_matcher_t::per_enc.at(encoding_t::UTYPE)},
         }},
        {encoding_t::JTYPE,
         {
             {reg_operand_info_t(true), reg_operand_matcher_t(0x00000f80, 7)},
             {imm_operand_info_t(),
              imm_operand_matcher_t::per_enc.at(encoding_t::JTYPE)},
         }},
        {encoding_t::SYSTYPE, {}},
        {encoding_t::CSRTYPE,
         {
             {reg_operand_info_t(true), reg_operand_matcher_t(0x00000f80, 7)},
             {reg_operand_info_t(false), reg_operand_matcher_t(0x000f8000, 15)},
             {csr_operand_info_t(), csr_operand_matcher_t()},
         }},
        {encoding_t::CSRITYPE,
         {
             {reg_operand_info_t(true), reg_operand_matcher_t(0x00000f80, 7)},
             {imm_operand_info_t(),
              imm_operand_matcher_t::per_enc.at(encoding_t::CSRITYPE)},
             {csr_operand_info_t(), csr_operand_matcher_t()},
         }},
};

inline constexpr std::string_view opc2str(opcode_t opc) {
  switch (opc) {
#define OPC_TO_STR_CASE(opcode)                                                \
  case opcode_t::opcode:                                                       \
    return #opcode

    OPC_TO_STR_CASE(ADD);
    OPC_TO_STR_CASE(SUB);
    OPC_TO_STR_CASE(OR);
    OPC_TO_STR_CASE(XOR);
    OPC_TO_STR_CASE(AND);
    OPC_TO_STR_CASE(SRL);
    OPC_TO_STR_CASE(SLL);
    OPC_TO_STR_CASE(SRA);
    OPC_TO_STR_CASE(ADDI);
    OPC_TO_STR_CASE(ORI);
    OPC_TO_STR_CASE(XORI);
    OPC_TO_STR_CASE(ANDI);
    OPC_TO_STR_CASE(SRLI);
    OPC_TO_STR_CASE(SLLI);
    OPC_TO_STR_CASE(SRAI);
    OPC_TO_STR_CASE(JAL);
    OPC_TO_STR_CASE(JALR);
    OPC_TO_STR_CASE(BEQ);
    OPC_TO_STR_CASE(BNE);
    OPC_TO_STR_CASE(BLT);
    OPC_TO_STR_CASE(BGE);
    OPC_TO_STR_CASE(BLTU);
    OPC_TO_STR_CASE(BGEU);
    OPC_TO_STR_CASE(LB);
    OPC_TO_STR_CASE(LH);
    OPC_TO_STR_CASE(LW);
    OPC_TO_STR_CASE(LBU);
    OPC_TO_STR_CASE(LHU);
    OPC_TO_STR_CASE(SB);
    OPC_TO_STR_CASE(SH);
    OPC_TO_STR_CASE(SW);
    OPC_TO_STR_CASE(SLT);
    OPC_TO_STR_CASE(SLTU);
    OPC_TO_STR_CASE(SLTI);
    OPC_TO_STR_CASE(SLTIU);
    OPC_TO_STR_CASE(LUI);
    OPC_TO_STR_CASE(AUIPC);
    OPC_TO_STR_CASE(FENCE);
    OPC_TO_STR_CASE(ECALL);
    OPC_TO_STR_CASE(EBREAK);
    OPC_TO_STR_CASE(CSRRW);
    OPC_TO_STR_CASE(CSRRS);
    OPC_TO_STR_CASE(CSRRC);
    OPC_TO_STR_CASE(CSRRWI);
    OPC_TO_STR_CASE(CSRRSI);
    OPC_TO_STR_CASE(CSRRCI);

#undef OPC_TO_STR_CASE
  }
  assert(false);
}

class decoded_inst_t final {
  opcode_t opcode;
  std::vector<operand_t> operands;

public:
  decoded_inst_t(opcode_t opc, std::ranges::input_range auto &&opers)
      : opcode(opc),
        operands(std::ranges::begin(opers), std::ranges::end(opers)) {}

  auto opc() const noexcept { return opcode; }

  auto &ops() const & noexcept { return operands; }

  void print(std::ostream &os) const {
    auto opc_str = opc2str(opcode);
    for (auto c : opc_str)
      os << static_cast<char>(std::tolower(c));

    for (auto &op : operands) {
      op.visit([&](reg_operand_t reg) { os << "\tx" << reg.get_reg(); },
               [&](imm_operand_t imm) { os << '\t' << imm.imm(); },
               [&](csr_operand_t csr) { os << "\tCSR"; });
    }
    os << std::endl;
  }
};

class opcode_matcher_t final {
  inst_t filter_mask;
  inst_t match_mask;

public:
  consteval opcode_matcher_t(inst_t inst) noexcept
      : filter_mask(~static_cast<inst_t>(0)), match_mask(inst) {}

  consteval opcode_matcher_t(opcode_encoding_t opcode, funct3 f3,
                             funct7 f7) noexcept
      : filter_mask(opcode.filter() | f3.filter() | f7.filter()),
        match_mask(opcode | f3 | f7) {}

  constexpr bool match(inst_t inst) const noexcept {
    return (inst & filter_mask) == match_mask;
  }

  static const std::unordered_map<opcode_t, opcode_matcher_t> per_opc;
};

inline const std::unordered_map<opcode_t, opcode_matcher_t>
    opcode_matcher_t::per_opc = {
#define PRS_REGISTER_OPCODE(opc, f3, f7)                                       \
  {opcode_t::opc, opcode_matcher_t(opcode_bits(opcode_t::opc), (f3), (f7))}

        // arithmetic
        PRS_REGISTER_OPCODE(ADD, 0b000u, 0b0000000u),
        PRS_REGISTER_OPCODE(SUB, 0b000u, 0b0100000u),
        PRS_REGISTER_OPCODE(OR, 0b110u, 0b0000000u),
        PRS_REGISTER_OPCODE(XOR, 0b100u, 0b0000000u),
        PRS_REGISTER_OPCODE(AND, 0b111u, 0b0000000u),
        PRS_REGISTER_OPCODE(SRL, 0b101u, 0b0000000u),
        PRS_REGISTER_OPCODE(SLL, 0b001u, 0b0000000u),
        PRS_REGISTER_OPCODE(SRA, 0b101u, 0b0100000u),
        // arithmetic with immediate
        PRS_REGISTER_OPCODE(ADDI, 0b000u, no_funct7),
        PRS_REGISTER_OPCODE(ORI, 0b110u, no_funct7),
        PRS_REGISTER_OPCODE(XORI, 0b100u, no_funct7),
        PRS_REGISTER_OPCODE(ANDI, 0b111u, no_funct7),
        PRS_REGISTER_OPCODE(SRLI, 0b101u, no_funct7),
        PRS_REGISTER_OPCODE(SLLI, 0b001u, no_funct7),
        PRS_REGISTER_OPCODE(SRAI, 0b101u, no_funct7),
        // jumps and calls
        PRS_REGISTER_OPCODE(JAL, no_funct3, no_funct7),
        PRS_REGISTER_OPCODE(JALR, 0b000u, no_funct7),
        PRS_REGISTER_OPCODE(BEQ, 0b000u, no_funct7),
        PRS_REGISTER_OPCODE(BNE, 0b001u, no_funct7),
        PRS_REGISTER_OPCODE(BLT, 0b100u, no_funct7),
        PRS_REGISTER_OPCODE(BGE, 0b101u, no_funct7),
        PRS_REGISTER_OPCODE(BLTU, 0b110u, no_funct7),
        PRS_REGISTER_OPCODE(BGEU, 0b111u, no_funct7),
        // loads/sotres
        PRS_REGISTER_OPCODE(LB, 0b000u, no_funct7),
        PRS_REGISTER_OPCODE(LH, 0b001u, no_funct7),
        PRS_REGISTER_OPCODE(LW, 0b010u, no_funct7),
        PRS_REGISTER_OPCODE(LBU, 0b100u, no_funct7),
        PRS_REGISTER_OPCODE(LHU, 0b101u, no_funct7),
        PRS_REGISTER_OPCODE(SB, 0b000u, no_funct7),
        PRS_REGISTER_OPCODE(SH, 0b001u, no_funct7),
        PRS_REGISTER_OPCODE(SW, 0b010u, no_funct7),
        // data flow
        PRS_REGISTER_OPCODE(SLT, 0b010u, 0b0000000u),
        PRS_REGISTER_OPCODE(SLTU, 0b011u, 0b0000000u),
        PRS_REGISTER_OPCODE(SLTI, 0b010u, no_funct7),
        PRS_REGISTER_OPCODE(SLTIU, 0b100u, no_funct7),
        // upper immediate
        PRS_REGISTER_OPCODE(LUI, no_funct3, no_funct7),
        PRS_REGISTER_OPCODE(AUIPC, no_funct3, no_funct7),
        // special ones
        PRS_REGISTER_OPCODE(FENCE, 0b010u, 0b0000000u),
        {opcode_t::ECALL, opcode_matcher_t(0b1110011)},
        {opcode_t::EBREAK,
         opcode_matcher_t(0b00000000000100000000000001110011)},
        // zicsr
        PRS_REGISTER_OPCODE(CSRRW, 0b001u, no_funct7),
        PRS_REGISTER_OPCODE(CSRRS, 0b010u, no_funct7),
        PRS_REGISTER_OPCODE(CSRRC, 0b011u, no_funct7),
        PRS_REGISTER_OPCODE(CSRRWI, 0b101u, no_funct7),
        PRS_REGISTER_OPCODE(CSRRSI, 0b110u, no_funct7),
        PRS_REGISTER_OPCODE(CSRRCI, 0b111u, no_funct7),
#undef PRS_REGISTER_OPCODE
};

inline auto matcher4opcode(opcode_t opcode) {
  auto &matchers = opcode_matcher_t::per_opc;
  auto it = matchers.find(opcode);
  assert(it != matchers.end());
  return it->second;
}

inline std::vector<operand_description_t>
descriptions4encoding(encoding_t enc) {
  auto &descs = operand_description_t::per_enc;
  auto it = descs.find(enc);
  assert(it != descs.end());
  return it->second;
}

class inst_description_t {
  opcode_t opc;
  opcode_matcher_t opc_matcher;
  std::vector<operand_description_t> op_descs;

public:
  inst_description_t(opcode_t opcode)
      : opc(opcode), opc_matcher(matcher4opcode(opcode)),
        op_descs(descriptions4encoding(encoding4opcode(opcode))) {}

  auto opcode() const noexcept { return opc; }

  bool match_opc(inst_t inst) const noexcept { return opc_matcher.match(inst); }

  std::vector<operand_t> operands(inst_t inst) const {
    std::vector<operand_t> operands;
    for (const auto &op_desc : op_descs) {
      op_desc.visit(
          [inst, &operands](reg_operand_info_t reg_info,
                            reg_operand_matcher_t reg_matcher) {
            operands.push_back(
                operand_t(reg_info, reg_matcher.extract_reg(inst)));
          },
          [inst, &operands](imm_operand_info_t imm_info,
                            const imm_operand_matcher_t &imm_matcher) {
            operands.push_back(
                operand_t(imm_info, imm_matcher.extract_imm(inst)));
          },
          [inst, &operands](csr_operand_info_t csr_info,
                            csr_operand_matcher_t csr_matcher) {
            operands.push_back(
                operand_t(csr_info, csr_matcher.extract_csr(inst)));
          },
          [](reg_operand_info_t, const imm_operand_matcher_t &) {
            assert(false);
          },
          [](reg_operand_info_t, csr_operand_matcher_t) { assert(false); },
          [](imm_operand_info_t, reg_operand_matcher_t) { assert(false); },
          [](imm_operand_info_t, csr_operand_matcher_t) { assert(false); },
          [](csr_operand_info_t, reg_operand_matcher_t) { assert(false); },
          [](csr_operand_info_t, const imm_operand_matcher_t &) {
            assert(false);
          });
    }
    return operands;
  }

  static const std::vector<inst_description_t> all_descs;
};

class inst_info_t : private std::vector<inst_description_t> {
public:
  inst_info_t() : vector(opcodes.begin(), opcodes.end()) {}

  using vector::begin;
  using vector::cbegin;
  using vector::cend;
  using vector::end;
  using vector::rbegin;
  using vector::rend;

  using vector::size;
};

inline const inst_info_t inst_info;

inline auto decode(inst_t inst) {
  auto it = std::ranges::find_if(
      inst_info, [inst](const auto &desc) { return desc.match_opc(inst); });
  assert(it != inst_info.end());
  return decoded_inst_t(it->opcode(), it->operands(inst));
}

enum class mem_op_size_t {
  BYTE,
  HALF,
  WORD,
  UBYTE,
  UHALF,
};

enum class syscall_num_t {
  CLOSE = 57,
  WRITE = 64,
  NEWFSTAT = 80,
  EXIT = 93,
  BRK = 214,
};

class simulator_t final {
  memory_t memory;
  reg_file_t regs;

  bool trap_ebreak = false;

  static constexpr auto add_op = std::plus{};
  static constexpr auto sub_op = std::minus{};
  static constexpr auto or_op = std::bit_or{};
  static constexpr auto xor_op = std::bit_xor{};
  static constexpr auto and_op = std::bit_and{};
  static constexpr auto srl_op = [](reg_t s1, reg_t s2) {
    return to_ureg(s1) >> s2;
  };
  static constexpr auto sll_op = [](reg_t s1, reg_t s2) {
    return to_ureg(s1) << s2;
  };
  static constexpr auto sra_op = [](reg_t s1, reg_t s2) { return s1 >> s2; };
  static constexpr auto slt_op = [](reg_t s1, reg_t s2) {
    return s1 < s2 ? 1 : 0;
  };
  static constexpr auto sltu_op = [](reg_t s1, reg_t s2) {
    return to_ureg(s1) < to_ureg(s2) ? 1 : 0;
  };

  static constexpr auto eq_op = std::equal_to{};
  static constexpr auto ne_op = std::not_equal_to{};
  static constexpr auto lt_op = std::less{};
  static constexpr auto ge_op = std::greater_equal{};
  static constexpr auto ltu_op = [](reg_t s1, reg_t s2) {
    return lt_op(to_ureg(s1), to_ureg(s2));
  };
  static constexpr auto geu_op = [](reg_t s1, reg_t s2) {
    return ge_op(to_ureg(s1), to_ureg(s2));
  };

  static constexpr size_t rd_idx = 0;
  static constexpr size_t rs1_idx = 1;
  static constexpr size_t rs2_idx = 2;
  static constexpr size_t rs1_no_rd_idx = 0;
  static constexpr size_t rs2_no_rd_idx = 1;

  static constexpr bool dst = true;
  static constexpr bool src = false;

  template <bool is_dst = src> static size_t extract_reg(operand_t op) {
    size_t reg;
    op.visit(
        [&reg](reg_operand_t reg_op) {
          assert(reg_op.is_dst() == is_dst);
          reg = reg_op.get_reg();
        },
        [](auto other_ops) { assert(false); });
    return reg;
  }

  static reg_t extract_imm(operand_t op) {
    reg_t imm;
    op.visit([&imm](imm_operand_t imm_op) { imm = imm_op.imm(); },
             [](auto other_ops) { assert(false); });
    return imm;
  }

  void exec_reg_arithmetic(std::span<const operand_t> ops,
                           std::invocable<reg_t, reg_t> auto &&operation) {
    assert(ops.size() == 3);
    auto rd_op = ops[rd_idx];
    auto rs1_op = ops[rs1_idx];
    auto rs2_op = ops[rs2_idx];
    auto rd = extract_reg<dst>(rd_op);
    auto rs1 = extract_reg(rs1_op);
    auto rs2 = extract_reg(rs2_op);
    auto rs1_val = regs.read(rs1);
    auto rs2_val = regs.read(rs2);
    auto rd_val = operation(rs1_val, rs2_val);
    regs.write(rd, rd_val);
    regs.set_pc(regs.get_pc() + pc_step);
  }

  void exec_imm_arithmetic(std::span<const operand_t> ops,
                           std::invocable<reg_t, reg_t> auto &&operation) {
    assert(ops.size() == 3);
    auto rd_op = ops[rd_idx];
    auto rs1_op = ops[rs1_idx];
    constexpr size_t imm_idx = 2;
    auto imm_op = ops[imm_idx];
    auto rd = extract_reg<dst>(rd_op);
    auto rs1 = extract_reg(rs1_op);
    auto imm = extract_imm(imm_op);
    auto rs1_val = regs.read(rs1);
    auto rd_val = operation(rs1_val, imm);
    regs.write(rd, rd_val);
    regs.set_pc(regs.get_pc() + pc_step);
  }

  reg_t load(addr_t addr, mem_op_size_t size) {
    reg_t data;
    switch (size) {
    default:
      assert(false);
    case mem_op_size_t::BYTE:
      data = static_cast<reg_t>(static_cast<byte_t>(memory.read_byte(addr)));
      break;
    case mem_op_size_t::HALF:
      data = static_cast<reg_t>(static_cast<half_t>(memory.read_half(addr)));
      break;
    case mem_op_size_t::WORD:
      data = static_cast<reg_t>(static_cast<word_t>(memory.read_word(addr)));
      break;
    case mem_op_size_t::UBYTE:
      data = to_reg(static_cast<ureg_t>(memory.read_byte(addr)));
      break;
    case mem_op_size_t::UHALF:
      data = to_reg(static_cast<ureg_t>(memory.read_half(addr)));
      break;
    }
    std::cerr << "[mem:0x" << std::hex << std::setw(8) << addr << "]"
              << " ==> " << "0x" << std::setw(8) << data << std::dec
              << std::endl;
    return data;
  }

  void exec_load(std::span<const operand_t> ops, mem_op_size_t size) {
    assert(ops.size() == 3);
    auto rd_op = ops[rd_idx];
    auto rs1_op = ops[rs1_idx];
    constexpr size_t imm_idx = 2;
    auto imm_op = ops[imm_idx];
    auto rd = extract_reg<dst>(rd_op);
    auto rs1 = extract_reg(rs1_op);
    auto imm = extract_imm(imm_op);
    auto addr = to_addr(regs.read(rs1) + imm);
    auto rd_val = load(addr, size);
    regs.write(rd, rd_val);
    regs.set_pc(regs.get_pc() + pc_step);
  }

  void store(addr_t addr, reg_t data, mem_op_size_t size) {
    switch (size) {
    default:
      assert(false);
    case mem_op_size_t::BYTE:
      memory.write_byte(addr, static_cast<ubyte_t>(data & ~ubyte_t{0}));
      break;
    case mem_op_size_t::HALF:
      memory.write_half(addr, static_cast<uhalf_t>(data & ~uhalf_t{0}));
      break;
    case mem_op_size_t::WORD:
      memory.write_word(addr, static_cast<uword_t>(data & ~uword_t{0}));
      break;
    }
  }

  void exec_store(std::span<const operand_t> ops, mem_op_size_t size) {
    assert(ops.size() == 3);
    auto rs1_op = ops[rs1_no_rd_idx];
    auto rs2_op = ops[rs2_no_rd_idx];
    constexpr size_t imm_idx = 2;
    auto imm_op = ops[imm_idx];
    auto rs1 = extract_reg(rs1_op);
    auto rs2 = extract_reg(rs2_op);
    auto imm = extract_imm(imm_op);
    auto addr = to_addr(regs.read(rs1) + imm);
    store(addr, regs.read(rs2), size);
    regs.set_pc(regs.get_pc() + pc_step);
  }

  void exec_branch(std::span<const operand_t> ops,
                   std::invocable<reg_t, reg_t> auto &&comparator) {
    assert(ops.size() == 3);
    auto rs1_op = ops[rs1_no_rd_idx];
    auto rs2_op = ops[rs2_no_rd_idx];
    constexpr size_t imm_idx = 2;
    auto imm_op = ops[imm_idx];
    auto rs1 = extract_reg(rs1_op);
    auto rs2 = extract_reg(rs2_op);
    auto imm = extract_imm(imm_op);
    auto rs1_val = regs.read(rs1);
    auto rs2_val = regs.read(rs2);
    auto taken = comparator(rs1_val, rs2_val);
    auto offset = taken ? imm : pc_step;
    regs.set_pc(regs.get_pc() + offset);
  }

  void exec_jal(std::span<const operand_t> ops) {
    assert(ops.size() == 2);
    auto rd_op = ops[rd_idx];
    constexpr size_t imm_idx = 1;
    auto imm_op = ops[imm_idx];
    auto rd = extract_reg<dst>(rd_op);
    auto imm = extract_imm(imm_op);
    auto rd_val = regs.get_pc() + pc_step;
    regs.write(rd, rd_val);
    regs.set_pc(regs.get_pc() + imm);
  }

  void exec_jalr(std::span<const operand_t> ops) {
    assert(ops.size() == 3);
    auto rd_op = ops[rd_idx];
    auto rs1_op = ops[rs1_idx];
    constexpr size_t imm_idx = 2;
    auto imm_op = ops[imm_idx];
    auto rd = extract_reg<dst>(rd_op);
    auto rs1 = extract_reg(rs1_op);
    auto imm = extract_imm(imm_op);
    auto rd_val = regs.get_pc() + pc_step;
    regs.write(rd, rd_val);
    auto rs1_val = regs.read(rs1);
    regs.set_pc(rs1_val + imm);
  }


  std::unordered_map<int, int> open_fds = {{0, 0}, {1, 1}, {2, 2}};

  void process_close() {
    auto guest_fd = regs.read(riscv_abi_reg_t::a0);
    auto fd_pos = open_fds.find(guest_fd);
    if (fd_pos == open_fds.end()) {
      regs.write(riscv_abi_reg_t::a0, -1);
      return;
    }
    open_fds.erase(fd_pos);
    regs.write(riscv_abi_reg_t::a0, 0);
  }

  void process_write() {
    auto guest_fd = regs.read(riscv_abi_reg_t::a0);
    auto host_fd = open_fds.at(guest_fd);
    auto addr = to_addr(regs.read(riscv_abi_reg_t::a1));
    auto len = regs.read(riscv_abi_reg_t::a2);
    auto written = 0;
    for (auto i : std::views::iota(0, len)) {
      char data = memory.read_byte(addr + i);
      written += write(host_fd, &data, 1);
    }
    regs.write(riscv_abi_reg_t::a0, written);
  }

  void process_newfstat() {
    regs.write(riscv_abi_reg_t::a0, -1); // unimplemented
  }

  void process_exit() {
    auto ret_val = regs.read(riscv_abi_reg_t::a0);
    std::cerr << "exit called. Exiting." << std::endl;
    exit(ret_val);
  }

  void process_brk() {
    auto addr = to_addr(regs.read(riscv_abi_reg_t::a0));
    memory.extend_upto(addr);
    regs.write(riscv_abi_reg_t::a0, 0);
  }

  void exec_ecall(std::span<const operand_t> ops) {
    assert(ops.empty());
    auto syscall_num = regs.read(riscv_abi_reg_t::a7);
    switch (static_cast<syscall_num_t>(syscall_num)) {
    default:
      std::cerr << "Unsupported ecall: " << syscall_num << std::endl;
      assert(false);
    case syscall_num_t::CLOSE:
      process_close();
      break;
    case syscall_num_t::WRITE:
      process_write();
      break;
    case syscall_num_t::NEWFSTAT:
      process_newfstat();
      break;
    case syscall_num_t::EXIT:
      process_exit();
      break;
    case syscall_num_t::BRK:
      process_brk();
      break;
    }
    regs.set_pc(regs.get_pc() + pc_step);
  }

  void exec_ebreak(std::span<const operand_t> ops) {
    assert(ops.empty());
    trap_ebreak = true;
  }

  void exec_fence(std::span<const operand_t>) {
    regs.set_pc(regs.get_pc() + pc_step);
  }

  void exec_lui(std::span<const operand_t> ops) {
    assert(ops.size() == 2);
    auto rd_op = ops[rd_idx];
    constexpr size_t imm_idx = 1;
    auto imm_op = ops[imm_idx];
    auto rd = extract_reg<dst>(rd_op);
    auto imm = extract_imm(imm_op);
    regs.write(rd, imm);
    regs.set_pc(regs.get_pc() + pc_step);
  }

  void exec_auipc(std::span<const operand_t> ops) {
    assert(ops.size() == 2);
    auto rd_op = ops[rd_idx];
    constexpr size_t imm_idx = 1;
    auto imm_op = ops[imm_idx];
    auto rd = extract_reg<dst>(rd_op);
    auto imm = extract_imm(imm_op);
    regs.write(rd, regs.get_pc() + imm);
    regs.set_pc(regs.get_pc() + pc_step);
  }

public:
  simulator_t() = default;

  inst_t fetch_inst() { return memory.read_word(regs.get_pc()); }

  void exec_one() {
    auto inst = fetch_inst();
    auto decoded_inst = decode(inst);
    std::cerr << "0x" << std::hex << std::setw(8) << std::setfill('0')
              << regs.get_pc() << ":\t" << std::dec;
    decoded_inst.print(std::cerr);
    switch (decoded_inst.opc()) {
    case opcode_t::ADD:
      exec_reg_arithmetic(decoded_inst.ops(), add_op);
      break;
    case opcode_t::SUB:
      exec_reg_arithmetic(decoded_inst.ops(), sub_op);
      break;
    case opcode_t::OR:
      exec_reg_arithmetic(decoded_inst.ops(), or_op);
      break;
    case opcode_t::XOR:
      exec_reg_arithmetic(decoded_inst.ops(), xor_op);
      break;
    case opcode_t::AND:
      exec_reg_arithmetic(decoded_inst.ops(), and_op);
      break;
    case opcode_t::SRL:
      exec_reg_arithmetic(decoded_inst.ops(), srl_op);
      break;
    case opcode_t::SLL:
      exec_reg_arithmetic(decoded_inst.ops(), sll_op);
      break;
    case opcode_t::SRA:
      exec_reg_arithmetic(decoded_inst.ops(), sra_op);
      break;
    case opcode_t::SLT:
      exec_reg_arithmetic(decoded_inst.ops(), slt_op);
      break;
    case opcode_t::SLTU:
      exec_reg_arithmetic(decoded_inst.ops(), sltu_op);
      break;
    case opcode_t::ADDI:
      exec_imm_arithmetic(decoded_inst.ops(), add_op);
      break;
    case opcode_t::ORI:
      exec_imm_arithmetic(decoded_inst.ops(), or_op);
      break;
    case opcode_t::XORI:
      exec_imm_arithmetic(decoded_inst.ops(), xor_op);
      break;
    case opcode_t::ANDI:
      exec_imm_arithmetic(decoded_inst.ops(), and_op);
      break;
    case opcode_t::SRLI:
      exec_imm_arithmetic(decoded_inst.ops(), srl_op);
      break;
    case opcode_t::SLLI:
      exec_imm_arithmetic(decoded_inst.ops(), sll_op);
      break;
    case opcode_t::SRAI:
      exec_imm_arithmetic(decoded_inst.ops(), sra_op);
      break;
    case opcode_t::JALR:
      exec_jalr(decoded_inst.ops());
      break;
    case opcode_t::LB:
      exec_load(decoded_inst.ops(), mem_op_size_t::BYTE);
      break;
    case opcode_t::LH:
      exec_load(decoded_inst.ops(), mem_op_size_t::HALF);
      break;
    case opcode_t::LW:
      exec_load(decoded_inst.ops(), mem_op_size_t::WORD);
      break;
    case opcode_t::LBU:
      exec_load(decoded_inst.ops(), mem_op_size_t::UBYTE);
      break;
    case opcode_t::LHU:
      exec_load(decoded_inst.ops(), mem_op_size_t::UHALF);
      break;
    case opcode_t::SLTI:
      exec_reg_arithmetic(decoded_inst.ops(), slt_op);
      break;
    case opcode_t::SLTIU:
      exec_reg_arithmetic(decoded_inst.ops(), sltu_op);
      break;
    case opcode_t::ECALL:
      exec_ecall(decoded_inst.ops());
      break;
    case opcode_t::EBREAK:
      exec_ebreak(decoded_inst.ops());
      break;
    case opcode_t::FENCE:
      exec_fence(decoded_inst.ops());
      break;
    case opcode_t::JAL:
      exec_jal(decoded_inst.ops());
      break;
    case opcode_t::BEQ:
      exec_branch(decoded_inst.ops(), eq_op);
      break;
    case opcode_t::BNE:
      exec_branch(decoded_inst.ops(), ne_op);
      break;
    case opcode_t::BLT:
      exec_branch(decoded_inst.ops(), lt_op);
      break;
    case opcode_t::BGE:
      exec_branch(decoded_inst.ops(), ge_op);
      break;
    case opcode_t::BLTU:
      exec_branch(decoded_inst.ops(), ltu_op);
      break;
    case opcode_t::BGEU:
      exec_branch(decoded_inst.ops(), geu_op);
      break;
    case opcode_t::SB:
      exec_store(decoded_inst.ops(), mem_op_size_t::BYTE);
      break;
    case opcode_t::SH:
      exec_store(decoded_inst.ops(), mem_op_size_t::HALF);
      break;
    case opcode_t::SW:
      exec_store(decoded_inst.ops(), mem_op_size_t::WORD);
      break;
    case opcode_t::LUI:
      exec_lui(decoded_inst.ops());
      break;
    case opcode_t::AUIPC:
      exec_auipc(decoded_inst.ops());
      break;
    default:
      assert(false);
    }
  }

  void exec_loop(addr_t start) {
    regs.set_pc(start);
    for (;;) {
      if (trap_ebreak)
        break;
      exec_one();
    }
  }

  void load_section(const char *data, size_t size, addr_t addr) {
    memory.load_section(data, size, addr);
  }

  void init_sp(reg_t sp) { regs.write(riscv_abi_reg_t::sp, sp); }
};

void run_sim(int argc, char *argv[]);
