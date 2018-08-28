//=======================================================================
// Copyright 2009 Trustees of Indiana University.
// Authors: Michael Hansen
//
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//=======================================================================

#include <fstream>
#include <iostream>
#include <string>

#include <boost/algorithm/string.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/filtered_graph.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/iteration_macros.hpp>
#include <boost/graph/mcgregor_common_subgraphs.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/property_map/shared_array_property_map.hpp>
#define MAX_TIME_S (25)

using namespace boost;

// Callback that looks for the first common subgraph whose size
// matches the user's preference.
template <typename Graph> struct example_callback {

    typedef typename graph_traits<Graph>::vertices_size_type VertexSizeFirst;

    example_callback(int *size, std::time_t start) : size(size), start(start)
    {
    }

    template <typename CorrespondenceMapFirstToSecond,
              typename CorrespondenceMapSecondToFirst>
    bool operator()(CorrespondenceMapFirstToSecond correspondence_map_1_to_2,
                    CorrespondenceMapSecondToFirst correspondence_map_2_to_1,
                    VertexSizeFirst subgraph_size)
    {

        (void)correspondence_map_1_to_2;
        (void)correspondence_map_2_to_1;

        if ((int)subgraph_size > *this->size) {
            *this->size = subgraph_size;
        }

        return std::difftime(std::time(nullptr), this->start) < MAX_TIME_S;
    }

  private:
    int *size;
    const std::time_t start;
};

extern "C" void subgraphs(char *nodes)
{

    // Using a vecS graph here so that we don't have to mess around with
    // a vertex index map; it will be implicit.
    typedef adjacency_list<listS, vecS, undirectedS,
                           property<vertex_name_t, unsigned int,
                                    property<vertex_index_t, unsigned int>>,
                           property<edge_name_t, unsigned int>>
            Graph;

    typedef property_map<Graph, vertex_name_t>::type VertexNameMap;

    // Test maximum and unique variants on known graphs
    Graph graph1, graph2;

    VertexNameMap vname_map1 = get(vertex_name, graph1);
    VertexNameMap vname_map2 = get(vertex_name, graph2);

    std::string line;
    bool reading_nodes = true;
    bool in_first = true;
    std::string s(nodes);
    std::istringstream f(s);
    while (getline(f, line, '|')) {
        std::vector<std::string> results;
        boost::split(results, line, [](char c) { return c == ','; });

        if (results.size() == 1 && !reading_nodes) {
            reading_nodes = true;
            in_first = false;
        } else if (results.size() == 2) {
            reading_nodes = false;
        }

        if (reading_nodes) {
            put(in_first ? vname_map1 : vname_map2,
                add_vertex(in_first ? graph1 : graph2), std::stoi(line));
        } else {
            add_edge(std::stoi(results[0]), std::stoi(results[1]),
                     in_first ? graph1 : graph2);
        }
    }

    int out = 0;
    example_callback<Graph> user_callback(&out, std::time(nullptr));

    // Maximum subgraphs
    mcgregor_common_subgraphs(graph1, graph2, true, user_callback);

    std::cout << out << '\t' << num_vertices(graph1) << '\t' << num_vertices(graph2) << std::endl;
}

int main(int argc, char *argv[])
{
    (void)argc;
    int end = strlen(argv[1]);
    while (argv[1][end] == '\n') {
        argv[1][end--] = '\0';
    }
    subgraphs(argv[1]);
    return 0;
}
