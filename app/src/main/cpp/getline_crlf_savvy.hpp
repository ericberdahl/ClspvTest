//
// Created by Eric Berdahl on 4/26/18.
//

#ifndef GETLINE_CRLF_SAVVY_HPP
#define GETLINE_CRLF_SAVVY_HPP

#include <ios>
#include <string>

namespace crlf_savvy {

    // Implementation of getline that recognizes CR, LF, and CRLF all as valid line endings,
    // allowing software to read text files that may have been written with different line endings
    // than that which is preferred by the host system on which the software is running.
    std::istream& getline(std::istream& is, std::string& t);

}

#endif //GETLINE_CRLF_SAVVY_HPP
