//
// Created by 16586 on 2022/3/30.
//

#include "../include/hard_api.h"
#include <iostream>

int main() {
  mpz_class message("0x9205993da57");

  std::cout << "Message: " << message.get_str(16) << std::endl;

  auto h = hard::Hard();


  mpz_class cipher = h.encrypt(message);

  std::cout << "Cipher is: " << cipher.get_str(16) << std::endl;

  auto de_message = h.decrypt(cipher);

  std::cout << "De_message is: " << de_message.get_str(16) << std::endl;

  auto hex_file = "/home/xilinx/temp/key.hex";
  h.load_param(hex_file);

  std::cout << "Param loaded" << std::endl;

  cipher = h.encrypt(message);
  std::cout << "Cipher is: " << cipher.get_str(16) << std::endl;
  de_message = h.decrypt(cipher);
  std::cout << "De_message is: " << de_message.get_str(16) << std::endl;

  return 0;
}