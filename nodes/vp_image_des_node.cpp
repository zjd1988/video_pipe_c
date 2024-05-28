

#include "vp_image_des_node.h"


namespace vp_nodes {
        
    vp_image_des_node::vp_image_des_node(std::string node_name, 
                                    int channel_index,  
                                    std::string location, 
                                    int interval,
                                    vp_objects::vp_size resolution_w_h,
                                    bool osd,
                                    std::string gst_encoder_name):
                                    vp_des_node(node_name, channel_index),
                                    location(location),
                                    interval(interval),
                                    resolution_w_h(resolution_w_h),
                                    osd(osd),
                                    gst_encoder_name(gst_encoder_name) {
        // make sure not greater than 1 minute (too long) and not lower than 1 second (since it's too quick, use video stream instead directly)
        assert(interval >= 1 && interval <= 60);
        if (vp_utils::ends_with(location, ".jpeg") || vp_utils::ends_with(location, ".jpg")) {
            // save to file
            gst_template_file = vp_utils::string_format(gst_template_file, interval, gst_encoder_name.c_str(), location.c_str());
            to_file = true;
        }
        else if (location.find(":") != std::string::npos) {
            // push to remote
            auto parts = vp_utils::string_split(location, ':');
            assert(parts.size() == 2);
            auto host = parts[0];  // ip
            auto port = std::stoi(parts[1]);  // try to get port

            gst_template_udp = vp_utils::string_format(gst_template_udp, interval, gst_encoder_name.c_str(), host.c_str(), port);
            
            to_file = false;
        }
        else {
            // error
            throw "invalid input parameter for `location`!";
        }

        auto s = to_file ? gst_template_file : gst_template_udp;
        VP_INFO(vp_utils::string_format("[%s] [%s]", node_name.c_str(), s.c_str()));
        this->initialized();
    }
    
    vp_image_des_node::~vp_image_des_node() {
        deinitialized();
    }

    std::shared_ptr<vp_objects::vp_meta> vp_image_des_node::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
        VP_DEBUG(vp_utils::string_format("[%s] received frame meta, channel_index=>%d, frame_index=>%d", node_name.c_str(), meta->channel_index, meta->frame_index));
        
        cv::Mat resize_frame;
        if (this->resolution_w_h.width != 0 && this->resolution_w_h.height != 0) {                 
            cv::resize((osd && !meta->osd_frame.empty()) ? meta->osd_frame : meta->frame, resize_frame, cv::Size(resolution_w_h.width, resolution_w_h.height));
        }
        else {
            resize_frame = (osd && !meta->osd_frame.empty()) ? meta->osd_frame : meta->frame;
        }

        if (!image_writer.isOpened()) {
            if (to_file) {
                assert(image_writer.open(this->gst_template_file, cv::CAP_GSTREAMER, 0, meta->fps, {resize_frame.cols, resize_frame.rows}));
            }
            else {
                assert(image_writer.open(this->gst_template_udp, cv::CAP_GSTREAMER, 0, meta->fps, {resize_frame.cols, resize_frame.rows}));
            }
        }

        image_writer.write(resize_frame);

        // for general works defined in base class
        return vp_des_node::handle_frame_meta(meta);
    }

    std::string vp_image_des_node::to_string() {
        return to_file ? location : "udp://" + location + "/jpg";
    }
}