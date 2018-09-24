//
// Created by Eric Berdahl on 4/26/18.
//
// Based on code found on Stack Overflow
// https://stackoverflow.com/questions/6089231/getting-std-ifstream-to-handle-lf-cr-and-crlf
//

#include "crlf_savvy.hpp"

#include <istream>

namespace crlf_savvy {

    std::istream& getline(std::istream &is, std::string &t) {
        std::streambuf* saveBuffer = is.rdbuf();

        crlf_filter_buffer filter(saveBuffer);
        is.rdbuf(&filter);

        std::getline(is, t);

        is.rdbuf(saveBuffer);

        return is;
    }

    crlf_filter_buffer::crlf_filter_buffer(std::streambuf* source)
            : mSource(source)
    {
        char *end = &mBuffer + 1;
        setg(end, end, end);
    }

    crlf_filter_buffer::int_type
    crlf_filter_buffer::underflow()
    {
        if (gptr() < egptr()) // buffer not exhausted
            return traits_type::to_int_type(*gptr());

        int_type peekInt = mSource->sbumpc();
        if (peekInt == traits_type::eof())
            return traits_type::eof();

        if (peekInt == '\r' && mSource->sgetc() == '\n')
        {
            peekInt = mSource->sbumpc();
        }

        mBuffer = traits_type::to_char_type(peekInt);

        char* base = &mBuffer;
        setg(base, base, base + 1);

        return traits_type::to_int_type(*gptr());
    }


}
