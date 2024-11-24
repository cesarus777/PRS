#include "elfio/elfio.hpp"
#include <elfio/elfio_symbols.hpp>
#include <limits>

#include "sim.hpp"

void run_sim(int argc, char *argv[]) {
  assert(argc == 2);
  auto *elf_path = argv[1];
  ELFIO::elfio reader;
  if (!reader.load(elf_path)) {
    std::cerr << "Can't open '" << elf_path << "'" << std::endl;
    return;
  }

  simulator_t sim;
  for (auto &&segment : reader.segments) {
    sim.load_section(segment->get_data(), segment->get_file_size(),
                     segment->get_virtual_address());
  }
  auto sym_sec = std::ranges::find_if(reader.sections, [](auto &sec) {
    return sec->get_type() == ELFIO::SHT_SYMTAB;
  });
  assert(sym_sec != reader.sections.end());
  ELFIO::symbol_section_accessor sym_table(reader, sym_sec->get());

  std::string name = "_start";
  ELFIO::Elf64_Addr value;
  ELFIO::Elf_Xword size;
  unsigned char bind;
  unsigned char type;
  ELFIO::Elf_Half section_index;
  unsigned char other;
  sym_table.get_symbol(name, value, size, bind, type, section_index, other);

  assert(value <= std::numeric_limits<addr_t>::max());

  sim.init_sp(0x100000);

  addr_t start = value;
  sim.exec_loop(start);
}
