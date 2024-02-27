#include <set>
#include "vp_yolo_tritonserver_node.h"

namespace vp_nodes {

    static std::string coco_cls_to_name(int cls_id, std::vector<std::string>& labels)
    {
        if (cls_id >= labels.size())
        {
            return "null";
        }
        if (labels[cls_id] != "")
        {
            return labels[cls_id];
        }
        return "null";
    }

    static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                                float ymax1)
    {
        float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
        float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
        float i = w * h;
        float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
        return u <= 0.f ? 0.f : (i / u);
    }

    static int nms(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds, std::vector<int> &order,
                int filterId, float threshold)
    {
        for (int i = 0; i < validCount; ++i)
        {
            if (order[i] == -1 || classIds[i] != filterId)
            {
                continue;
            }
            int n = order[i];
            for (int j = i + 1; j < validCount; ++j)
            {
                int m = order[j];
                if (m == -1 || classIds[i] != filterId)
                {
                    continue;
                }
                float xmin0 = outputLocations[n * 4 + 0];
                float ymin0 = outputLocations[n * 4 + 1];
                float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
                float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

                float xmin1 = outputLocations[m * 4 + 0];
                float ymin1 = outputLocations[m * 4 + 1];
                float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
                float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

                float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

                if (iou > threshold)
                {
                    order[j] = -1;
                }
            }
        }
        return 0;
    }

    static int quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices)
    {
        float key;
        int key_index;
        int low = left;
        int high = right;
        if (left < right)
        {
            key_index = indices[left];
            key = input[left];
            while (low < high)
            {
                while (low < high && input[high] <= key)
                {
                    high--;
                }
                input[low] = input[high];
                indices[low] = indices[high];
                while (low < high && input[low] >= key)
                {
                    low++;
                }
                input[high] = input[low];
                indices[high] = indices[low];
            }
            input[low] = key;
            indices[low] = key_index;
            quick_sort_indice_inverse(input, left, low - 1, indices);
            quick_sort_indice_inverse(input, low + 1, right, indices);
        }
        return low;
    }

    static int process(float *input, int *anchor, int grid_h, int grid_w, int height, int width, int stride, 
        int prop_box_size, int obj_class_num, std::vector<float> &boxes, std::vector<float> &objProbs, 
        std::vector<int> &classId, float threshold)
    {
        int validCount = 0;
        int grid_len = grid_h * grid_w;

        for (int a = 0; a < 3; a++)
        {
            for (int i = 0; i < grid_h; i++)
            {
                for (int j = 0; j < grid_w; j++)
                {
                    float box_confidence = input[(prop_box_size * a + 4) * grid_len + i * grid_w + j];
                    if (box_confidence >= threshold)
                    {
                        int offset = (prop_box_size * a) * grid_len + i * grid_w + j;
                        float *in_ptr = input + offset;
                        float box_x = *in_ptr * 2.0 - 0.5;
                        float box_y = in_ptr[grid_len] * 2.0 - 0.5;
                        float box_w = in_ptr[2 * grid_len] * 2.0;
                        float box_h = in_ptr[3 * grid_len] * 2.0;
                        box_x = (box_x + j) * (float)stride;
                        box_y = (box_y + i) * (float)stride;
                        box_w = box_w * box_w * (float)anchor[a * 2];
                        box_h = box_h * box_h * (float)anchor[a * 2 + 1];
                        box_x -= (box_w / 2.0);
                        box_y -= (box_h / 2.0);

                        float maxClassProbs = in_ptr[5 * grid_len];
                        int maxClassId = 0;
                        for (int k = 1; k < obj_class_num; ++k)
                        {
                            float prob = in_ptr[(5 + k) * grid_len];
                            if (prob > maxClassProbs)
                            {
                                maxClassId = k;
                                maxClassProbs = prob;
                            }
                        }
                        if (maxClassProbs > threshold)
                        {
                            objProbs.push_back(maxClassProbs * box_confidence);
                            classId.push_back(maxClassId);
                            validCount++;
                            boxes.push_back(box_x);
                            boxes.push_back(box_y);
                            boxes.push_back(box_w);
                            boxes.push_back(box_h);
                        }
                    }
                }
            }
        }
        return validCount;
    }

    vp_yolo_tritonserver_node::vp_yolo_tritonserver_node(std::string node_name, 
                            std::string model_path, 
                            std::string model_name, 
                            int         model_version, 
                            std::string model_config_path, 
                            std::string labels_path, 
                            int input_width, 
                            int input_height, 
                            int batch_size,
                            int class_id_offset,
                            float score_threshold,
                            float confidence_threshold,
                            float nms_threshold,
                            float scale,
                            cv::Scalar mean,
                            cv::Scalar std,
                            bool swap_rb):
                            vp_primary_infer_node(node_name, model_path, model_name, model_version, model_config_path, 
                                labels_path, input_width, input_height, batch_size, class_id_offset, scale, mean, std, swap_rb),
                            score_threshold(score_threshold),
                            confidence_threshold(confidence_threshold),
                            nms_threshold(nms_threshold) {
        this->initialized();
    }
    
    vp_yolo_tritonserver_node::~vp_yolo_tritonserver_node() {

    }

    void vp_yolo_tritonserver_node::preprocess(const std::vector<cv::Mat>& mats_to_infer, cv::Mat& blob_to_infer)
    {
        cv::dnn::blobFromImages(mats_to_infer, blob_to_infer, scale, cv::Size(input_width, input_height), mean, swap_rb);
    }
    
    void vp_yolo_tritonserver_node::postprocess(const std::vector<cv::Mat>& raw_outputs, 
        const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch)
    {
        auto& frame_meta = frame_meta_with_batch[0];
        std::vector<float> filterBoxes;
        std::vector<float> objProbs;
        std::vector<int> classId;
        int validCount = 0;
        int stride[] = {8, 16, 32};
        int grid_h[] = {80, 40, 20};
        int grid_w[] = {80, 40, 20};
        int model_in_w = input_width;
        int model_in_h = input_height;
        int anchor[3][6] = {{10, 13, 16, 30, 33, 23},
                            {30, 61, 62, 45, 59, 119},
                            {116, 90, 156, 198, 373, 326}};

        int obj_class_num = labels.size();
        for (int i = 0; i < 3; i++)
        {
            int prop_box_size = raw_outputs[i].size[1] / 3;
            validCount += process((float *)raw_outputs[i].data, (int *)anchor[i], grid_h[i], 
                grid_w[i], model_in_h, model_in_w, stride[i], prop_box_size, obj_class_num, filterBoxes, 
                objProbs, classId, confidence_threshold);
        }

        // no object detect
        if (validCount <= 0)
        {
            return;
        }
        std::vector<int> indexArray;
        for (int i = 0; i < validCount; ++i)
        {
            indexArray.push_back(i);
        }
        quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

        std::set<int> class_set(std::begin(classId), std::end(classId));

        for (auto c : class_set)
        {
            nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
        }

        /* box valid detect target */
        for (int i = 0; i < validCount; ++i)
        {
            if (indexArray[i] == -1)
            {
                continue;
            }
            int n = indexArray[i];

            float x1 = filterBoxes[n * 4 + 0];
            float y1 = filterBoxes[n * 4 + 1];
            float x2 = x1 + filterBoxes[n * 4 + 2];
            float y2 = y1 + filterBoxes[n * 4 + 3];
            int id = classId[n];
            float obj_conf = objProbs[i];
            std::string label = coco_cls_to_name(id, labels);
            auto target = std::make_shared<vp_objects::vp_frame_target>(x1, y1, x2 - x1, 
                y2 - y1, id, obj_conf, frame_meta->frame_index, frame_meta->channel_index, label);

            // insert target back to frame meta
            frame_meta->targets.push_back(target);
        }
        return;
    }
}