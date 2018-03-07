/*****************************************************************************
 * This file is part of the Project SchizophrenicQuicksort
 * 
 * Copyright (c) 2016-2017, Tobias Heuer <tobias.heuer@gmx.net>
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 * Copyright (c) 2016-2017, Michael Axtmann <michael.axtmann@kit.edu>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
******************************************************************************/

#ifndef SORTINGDATATYPE_H
#define SORTINGDATATYPE_H

#include <mpi.h>
#include <type_traits>

enum class SORT_TYPE : uint8_t {
    Int, Double, Long, Float, Pair
};

//#pragma pack(4)
struct sort_pair {
	float value;
	int32_t index;
	bool operator<(const sort_pair& rhs ) const {return value < rhs.value || (value == rhs.value && index < rhs.index);}
	bool operator>(const sort_pair& rhs ) const {return value > rhs.value || (value == rhs.value && index > rhs.index);}
	bool operator<=(const sort_pair& rhs ) const {return !operator>(rhs);}
	bool operator>=(const sort_pair& rhs ) const {return !operator<(rhs);}
};

template<typename T>
class SortingDatatype {
public:

    static MPI_Datatype getMPIDatatype() {
        if (std::is_same<T, int>::value)
            return MPI_INT;
        else if (std::is_same<T, short>::value)
            return MPI_SHORT;
        else if (std::is_same<T, char>::value)
            return MPI_CHAR;
        else if (std::is_same<T, long>::value)
            return MPI_LONG;
        else if (std::is_same<T, float>::value)
            return MPI_FLOAT;
        else if (std::is_same<T, double>::value)
            return MPI_DOUBLE;
        else if (std::is_same<T, sort_pair>::value)
            return MPI_FLOAT_INT;
    }

    static SORT_TYPE getSortType(std::string datatype) {
        if (datatype.compare("int") == 0)
            return SORT_TYPE::Int;
        else if (datatype.compare("double") == 0)
            return SORT_TYPE::Double;
        else if (datatype.compare("long") == 0)
            return SORT_TYPE::Long;
        else if (datatype.compare("float") == 0)
            return SORT_TYPE::Float;
        else if (datatype.compare("struct-float-int") == 0)
            return SORT_TYPE::Pair;
        else
            return SORT_TYPE::Int;
    }
    
    static std::string getString() {
        if (std::is_same<T, int>::value)
            return "int";
        else if (std::is_same<T, short>::value)
            return "short";
        else if (std::is_same<T, char>::value)
            return "char";
        else if (std::is_same<T, long>::value)
            return "long";
        else if (std::is_same<T, float>::value)
            return "float";
        else if (std::is_same<T, double>::value)
            return "double";
        else if (std::is_same<T, sort_pair>::value)
            return "struct-float-int";
    }
   
};

#endif /* SORTINGDATATYPE_H */
