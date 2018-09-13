//
// Created by Eric Berdahl on 4/26/18.
//

#ifndef CLSPVUTILS_GETLINE_CRLF_SAVVY_HPP
#define CLSPVUTILS_GETLINE_CRLF_SAVVY_HPP

#include <ios>
#include <string>

namespace crlf_savvy {

    // Implementation of getline that recognizes CR, LF, and CRLF all as valid line endings,
    // allowing software to read text files that may have been written with different line endings
    // than that which is preferred by the host system on which the software is running.
    std::istream& getline(std::istream& is, std::string& t);

}

#endif //CLSPVUTILS_GETLINE_CRLF_SAVVY_HPP
