// This file is part of the AliceVision project.
// Copyright (c) 2023 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include "msgpackIO.hpp"

#include <msgpack.hpp>

#include <sstream>
#include <string>

MSGPACK_ADD_ENUM(aliceVision::feature::EImageDescriberType);


namespace msgpack {
MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS) {
namespace adaptor {

template <class Key ,class T>
struct convert<stl::flat_map<Key, T>> {
    msgpack::object const& operator()(const msgpack::object &o, stl::flat_map<Key, T> & output) const 
    {
        if (o.type != msgpack::type::ARRAY) throw msgpack::type_error();
        
        if (o.via.array.size != 1) throw msgpack::type_error();

        std::vector<std::pair<Key, T>> observations = o.via.array.ptr[0].as<std::vector<std::pair<Key, T>>>();

        for (auto item : observations)
        {
            output.insert(item);
        }

        return o;
    }
};

template <class Key ,class T>
struct pack<stl::flat_map<Key, T>> {
    template <typename Stream>
    msgpack::packer<Stream>& operator()(msgpack::packer<Stream>& o, stl::flat_map<Key, T> const& input) const {

        std::vector<std::pair<Key, T>> vector;
        for (const auto & pair : input)
        {
            vector.push_back(pair);
        }
        
        o.pack(vector);

        return o;
    }
};

template <>
struct convert<aliceVision::Vec2> {
    msgpack::object const& operator()(const msgpack::object & o, aliceVision::Vec2 & output) const 
    {
        if (o.type != msgpack::type::ARRAY) throw msgpack::type_error();
        
        if (o.via.array.size != 2) throw msgpack::type_error();

        output[0] = o.via.array.ptr[0].as<double>();
        output[1] = o.via.array.ptr[1].as<double>();
        
        return o;
    }
};

template <>
struct pack<aliceVision::Vec2> {
    template <typename Stream>
    msgpack::packer<Stream>& operator()(msgpack::packer<Stream>& o, aliceVision::Vec2 const& input) const {

        o.pack_array(2);
        o.pack(input(0));
        o.pack(input(1));

        return o;
    }
};

template <>
struct convert<aliceVision::Vec3> {
    msgpack::object const& operator()(const msgpack::object & o, aliceVision::Vec3 & output) const 
    {
        if (o.type != msgpack::type::ARRAY) throw msgpack::type_error();
        
        if (o.via.array.size != 3) throw msgpack::type_error();

        output[0] = o.via.array.ptr[0].as<double>();
        output[1] = o.via.array.ptr[1].as<double>();
        output[2] = o.via.array.ptr[2].as<double>();
        
        return o;
    }
};

template <>
struct pack<aliceVision::Vec3> {
    template <typename Stream>
    msgpack::packer<Stream>& operator()(msgpack::packer<Stream>& o, aliceVision::Vec3 const& input) const {

        o.pack_array(3);
        o.pack(input(0));
        o.pack(input(1));
        o.pack(input(2));

        return o;
    }
};

template <>
struct convert<aliceVision::Mat3> {
    msgpack::object const& operator()(const msgpack::object & o, aliceVision::Mat3 & output) const 
    {
        if (o.type != msgpack::type::ARRAY) throw msgpack::type_error();
        
        if (o.via.array.size != 9) throw msgpack::type_error();

        output(0, 0) = o.via.array.ptr[0].as<double>();
        output(0, 1) = o.via.array.ptr[1].as<double>();
        output(0, 2) = o.via.array.ptr[2].as<double>();

        output(1, 0) = o.via.array.ptr[3].as<double>();
        output(1, 1) = o.via.array.ptr[4].as<double>();
        output(1, 2) = o.via.array.ptr[5].as<double>();

        output(6, 0) = o.via.array.ptr[6].as<double>();
        output(7, 1) = o.via.array.ptr[7].as<double>();
        output(8, 2) = o.via.array.ptr[8].as<double>();
        
        return o;
    }
};

template <>
struct convert<aliceVision::image::RGBColor> {
    msgpack::object const& operator()(const msgpack::object & o, aliceVision::image::RGBColor & output) const 
    {
        if (o.type != msgpack::type::ARRAY) throw msgpack::type_error();
        
        if (o.via.array.size != 3) throw msgpack::type_error();

        output[0] = o.via.array.ptr[0].as<double>();
        output[1] = o.via.array.ptr[1].as<double>();
        output[2] = o.via.array.ptr[2].as<double>();
        
        return o;
    }
};

template <>
struct pack<aliceVision::image::RGBColor> {
    template <typename Stream>
    msgpack::packer<Stream>& operator()(msgpack::packer<Stream>& o, aliceVision::image::RGBColor const& input) const {

        o.pack_array(3);
        o.pack(input(0));
        o.pack(input(1));
        o.pack(input(2));

        return o;
    }
};


template <>
struct convert<aliceVision::geometry::Pose3> {
    msgpack::object const& operator()(const msgpack::object &o, aliceVision::geometry::Pose3 & output) const 
    {
        if (o.type != msgpack::type::ARRAY) throw msgpack::type_error();
        
        if (o.via.array.size != 2) throw msgpack::type_error();

        output = aliceVision::geometry::Pose3(
            o.via.array.ptr[0].as<aliceVision::Mat3>(),
            o.via.array.ptr[1].as<aliceVision::Vec3>()
        );
        
        return o;
    }
};


template <>
struct convert<aliceVision::sfmData::Observation> {
    msgpack::object const& operator()(const msgpack::object &o, aliceVision::sfmData::Observation & output) const 
    {
        if (o.type != msgpack::type::ARRAY) throw msgpack::type_error();
        
        if (o.via.array.size != 3) throw msgpack::type_error();

        output.x = o.via.array.ptr[0].as<aliceVision::Vec2>();
        output.id_feat = o.via.array.ptr[1].as<aliceVision::IndexT>();
        output.scale = o.via.array.ptr[2].as<double>();
        
        return o;
    }
};

template <>
struct pack<aliceVision::sfmData::Observation> {
    template <typename Stream>
    msgpack::packer<Stream>& operator()(msgpack::packer<Stream>& o, aliceVision::sfmData::Observation const& input) const {

        
        o.pack_array(4);
        o.pack(input.x(0));
        o.pack(input.x(1));
        o.pack(input.id_feat);
        o.pack(input.scale);

        return o;
    }
};

template <>
struct convert<aliceVision::sfmData::Landmark> {
    msgpack::object const& operator()(const msgpack::object &o, aliceVision::sfmData::Landmark & output) const 
    {
        if (o.type != msgpack::type::ARRAY) throw msgpack::type_error();
        
        if (o.via.array.size != 4) throw msgpack::type_error();

        output.X = o.via.array.ptr[0].as<aliceVision::Vec3>();
        output.rgb = o.via.array.ptr[1].as<aliceVision::image::RGBColor>();
        output.descType = o.via.array.ptr[2].as<aliceVision::feature::EImageDescriberType>();
        output.observations = o.via.array.ptr[2].as<aliceVision::sfmData::Observations>();
        
        return o;
    }
};

template <>
struct pack<aliceVision::sfmData::Landmark> {
    template <typename Stream>
    msgpack::packer<Stream>& operator()(msgpack::packer<Stream>& o, aliceVision::sfmData::Landmark const& l) const {

        
        o.pack_array(4);
        o.pack(l.X);
        o.pack(l.rgb);
        o.pack(l.descType);
        o.pack(l.observations);
        

        return o;
    }
};


template <>
struct convert<aliceVision::sfmData::CameraPose> {
    msgpack::object const& operator()(const msgpack::object &o, aliceVision::sfmData::CameraPose & output) const 
    {
        if (o.type != msgpack::type::ARRAY) throw msgpack::type_error();
        
        if (o.via.array.size != 2) throw msgpack::type_error();

        output = aliceVision::sfmData::CameraPose(
            o.via.array.ptr[0].as<aliceVision::geometry::Pose3>(),
            o.via.array.ptr[1].as<bool>()
        );
        
        return o;
    }
};

template <>
struct convert<aliceVision::sfmData::View> {
    msgpack::object const& operator()(const msgpack::object &o, aliceVision::sfmData::View & output) const 
    {
        if (o.type != msgpack::type::ARRAY) throw msgpack::type_error();
        
        if (o.via.array.size != 2) throw msgpack::type_error();

        output = aliceVision::sfmData::View(
            
        );
        
        return o;
    }
};

template <>
struct convert<aliceVision::sfmData::View, std::string> {
    msgpack::object const& operator()(const msgpack::object &o, aliceVision::sfmData::View & output) const 
    {
        if (o.type != msgpack::type::ARRAY) throw msgpack::type_error();
        
        if (o.via.array.size != 2) throw msgpack::type_error();

        output = aliceVision::sfmData::View(
            
        );
        
        return o;
    }
};

template <>
struct convert<aliceVision::sfmData::SfMData> {
    msgpack::object const& operator()(const msgpack::object &o, aliceVision::sfmData::SfMData & output) const 
    {
        if (o.type != msgpack::type::MAP) throw msgpack::type_error();
        
        auto map = o.as<std::map<std::string, msgpack::object>>();

        aliceVision::sfmData::Views & views = output.getViews();
        views = map["views"].as<aliceVision::sfmData::Views>("tata");
        
        return o;
    }
};

template <>
struct pack<aliceVision::sfmData::SfMData> {
    template <typename Stream>
    msgpack::packer<Stream>& operator()(msgpack::packer<Stream>& o, aliceVision::sfmData::SfMData const& v) const {

        
        o.pack(v.getLandmarks());

        return o;
    }
};


} // namespace adaptor
} // MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
} // namespace msgpack

namespace aliceVision {
namespace sfmDataIO {

bool saveMsgPack(const sfmData::SfMData& sfmData, const std::string& filename, ESfMData partFlag)
{
    std::ofstream f(filename);

    msgpack::pack(f, sfmData);

    return true;
}

bool loadMsgPack(sfmData::SfMData& sfmData, const std::string& filename, ESfMData partFlag)
{
    return true;
}

} // namespace sfmDataIO
} // namespace aliceVision