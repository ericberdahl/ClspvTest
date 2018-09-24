//
// Created by Eric Berdahl on 4/26/18.
//

#ifndef CLSPVUTILS_CRLF_SAVVY_HPP
#define CLSPVUTILS_CRLF_SAVVY_HPP

#include <iosfwd>
#include <streambuf>
#include <string>

namespace crlf_savvy {

    // Implementation of getline that recognizes CR, LF, and CRLF all as valid line endings,
    // allowing software to read text files that may have been written with different line endings
    // than that which is preferred by the host system on which the software is running.
    std::istream& getline(std::istream& is, std::string& t);

    class crlf_filter_buffer : public std::streambuf
    {
    public:
        explicit            crlf_filter_buffer(std::streambuf* source);

        virtual int_type    underflow();

    private:
        std::streambuf* mSource;
        char            mBuffer;
    };
}

#endif //CLSPVUTILS_CRLF_SAVVY_HPP
