#include <open_ptrack/nms/nms.h>

class ulsMatF{
public:
    float* m_data;
    int m_rows, m_cols, m_channels;

public:
    ulsMatF(int cols, int rows, int channels){
        m_rows = rows;
        m_cols = cols;
        m_channels = channels;
        int size = channels * rows * cols * sizeof(float);
        m_data = (float*)malloc(size);
        memset((void *)m_data, 0, size);
    }
    // this during free tvmfree while a pointer causes some weird probs
    //~ulsMatF(){
    //    if(m_data) free(m_data);
    //}

    float *at(int channel, int row, int col){
        assert(m_data != NULL);
        assert(row < m_rows);
        assert(col < m_cols);
        assert(channel < m_channels);

        return m_data + (channel * m_rows * m_cols) + row * m_cols + col;
    }

    int getRows() {return m_rows;}
    int getCols() {return m_cols;}
    int getChannels() {return m_channels;}
};


class sortable_result {
public:
	sortable_result() {
	}

	~sortable_result() {
	}

    bool operator<(const sortable_result &t) const {
        return probs < t.probs;
    }

    bool operator>(const sortable_result &t) const {
        return probs > t.probs;
    }

    float& operator[](int i) {
        assert(0 <= i && i <= 4);

        if (i == 0) 
            return xmin;
        if (i == 1) 
            return ymin;
        if (i == 2) 
            return xmax;
        if (i == 3) 
            return ymax;
        return xmin;
    }

    float operator[](int i) const {
        assert(0 <= i && i <= 4);

        if (i == 0) 
            return xmin;
        if (i == 1) 
            return ymin;
        if (i == 2) 
            return xmax;
        if (i == 3) 
            return ymax;
        else{
            return 0;
        }
    }

    int index;
    int cls;
    float probs;
    float xmin;
    float ymin;
    float xmax;
    float ymax;

};

void nms_cpu(std::vector<sortable_result>& boxes, ulsMatF tvm_output, float cls_threshold, float nms_threshold, std::vector<sortable_result>& filterOutBoxes) {

    //ulsMatF(int cols, int rows, int channels)
    //at(int channel, int row, int col)
    //6, 322560, 1
    int valid_count = 0;
    for(int i = 0; i < tvm_output.getRows(); ++i){
      if (*tvm_output.at(0, i, 1) >= cls_threshold){
        sortable_result res;
        res.index = valid_count;
        res.cls = *tvm_output.at(0, i, 0);//tvm_output.at(0, i, 0);
        res.probs = *tvm_output.at(0, i, 1);//tvm_output.at(1, i, 0);
        res.xmin = *tvm_output.at(0, i, 2);//tvm_output.at(2, i, 0);
        res.ymin = *tvm_output.at(0, i, 3);//tvm_output.at(3, i, 0);
        res.xmax = *tvm_output.at(0, i, 4);//tvm_output.at(4, i, 0);
        res.ymax = *tvm_output.at(0, i, 5);//tvm_output.at(5, i, 0);
        boxes.push_back(res);
        valid_count+=1;
      }
    }
    std::cout << "valid count: " << valid_count << std::endl;
    filterOutBoxes.clear();
    if(boxes.size() == 0)
        return;
    std::vector<size_t> idx(boxes.size());

    for(unsigned i = 0; i < idx.size(); i++)
    {
        idx[i] = i;
    }

    //descending sort
    sort(boxes.begin(), boxes.end(), std::greater<sortable_result>());

    while(idx.size() > 0)
    {
        int good_idx = idx[0];
        filterOutBoxes.push_back(boxes[good_idx]);

        std::vector<size_t> tmp = idx;
        idx.clear();
        for(unsigned i = 1; i < tmp.size(); i++)
        {
            int tmp_i = tmp[i];
            float inter_x1 = std::max( boxes[good_idx].xmin, boxes[tmp_i].xmin );
            float inter_y1 = std::max( boxes[good_idx].ymin, boxes[tmp_i].ymin );
            float inter_x2 = std::min( boxes[good_idx].ymin, boxes[tmp_i].ymin );
            float inter_y2 = std::min( boxes[good_idx].ymax, boxes[tmp_i].ymax );

            float w = std::max((inter_x2 - inter_x1 + 1), 0.0F);
            float h = std::max((inter_y2 - inter_y1 + 1), 0.0F);

            float inter_area = w * h;
            float area_1 = (boxes[good_idx].xmax - boxes[good_idx].xmin + 1) * (boxes[good_idx].ymax - boxes[good_idx].ymin + 1);
            float area_2 = (boxes[tmp_i].xmax - boxes[tmp_i].xmin + 1) * (boxes[tmp_i].ymax - boxes[tmp_i].ymin + 1);
            float o = inter_area / (area_1 + area_2 - inter_area);           
            if( o <= nms_threshold )
                idx.push_back(tmp_i);
        }
    }
}