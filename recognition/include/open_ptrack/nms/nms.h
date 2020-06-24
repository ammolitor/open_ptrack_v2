#include <vector>
#include <iostream>


class MatF{
public:
    float* m_data;
    int m_rows, m_cols, m_channels;

public:
    MatF(int cols, int rows, int channels){
        m_rows = rows;
        m_cols = cols;
        m_channels = channels;
        int size = channels * rows * cols * sizeof(float);
        m_data = (float*)malloc(size);
        memset((void *)m_data, 0, size);
    }
    // this during free tvmfree while a pointer causes some weird probs
    //~MatF(){
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

void tvm_nms_cpu(std::vector<sortable_result>& boxes, MatF tvm_output, float cls_threshold, float nms_threshold, std::vector<sortable_result>& filterOutBoxes) {
