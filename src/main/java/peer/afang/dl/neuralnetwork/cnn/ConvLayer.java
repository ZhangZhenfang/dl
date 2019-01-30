package peer.afang.dl.neuralnetwork.cnn;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * @author ZhangZhenfang
 * @date 2019/1/28 9:57
 */
public class ConvLayer {

    /**
     * 过滤器大小
     */
    private int filterSize;
    /**
     * padding size
     */
    private int padding;
    /**
     * 卷积步长
     */
    private int stride;
    /**
     * 过滤器个数
     */
    private int numberOfFilters;

    private List<Mat> input;
    private List<List<Mat>> filters;
    private List<Mat> out;
    private List<Mat> delta;

    double[][][] testdata = new double[][][]{{{-1, 1, 0, 0, 1, 0, 0, 1, 1}, {-1, -1, 0, 0, 0, 0, 0, -1, 0}, {0, 0, -1, 0, 1, 0, 1, -1 , -1}}, {{1, 1, -1, -1, -1, 1, 0, -1, 1}, {0, 1, 0, -1, 0, -1, -1, 1, 0}, {-1, 0, 0, -1, 0, 1, -1, 0, 0}}};
    public ConvLayer() {

    }
    public ConvLayer(List<Mat> input, int filterSize, int padding, int stride, int numberOfFilters) {
        this.input = input;
        this.filterSize = filterSize;
        this.padding = padding;
        this.stride = stride;
        this.numberOfFilters = numberOfFilters;
        init();
    }

    private void init() {
        filters = new ArrayList<List<Mat>>();
        delta = new ArrayList<Mat>();
        out = new ArrayList<Mat>();
        int convSize = (input.get(0).rows() + 2 * padding - filterSize) / stride + 1;
        Random random = new Random();
        int length = filterSize * filterSize;
        double[] data = new double[length];

        for (int i = 0; i < numberOfFilters; i++) {
            List<Mat> mats = new ArrayList<Mat>();
            for (int j = 0; j < input.size(); j++) {
                Mat m = new Mat(filterSize, filterSize, input.get(0).type());
                for (int k = 0; k < length; k++) {
                    data[k] = (random.nextDouble() - 0.5) * 2;
                }
//                m.put(0, 0, testdata[i][j]);
                m.put(0, 0, data);
                mats.add(m);
            }
            filters.add(mats);
            out.add(Mat.zeros(convSize, convSize, CvType.CV_32F));
        }
    }

    public void computeOut() {
        for (int i = 0; i < filters.size(); i++) {
            Mat dst = new Mat(out.get(i).size(), out.get(i).type());
            for (int j = 0; j < filters.get(i).size(); j++) {
                Imgproc.filter2D(input.get(j), dst, input.get(j).depth(), filters.get(i).get(j));
                Core.add(dst, out.get(i), out.get(i));
            }
            Cnn.relu(out.get(i));
        }
    }

    public void computeDeltaFromFC(List<Mat> lastDelta, List<Mat> position) {
        for (int i = 0; i < lastDelta.size(); i++) {
            Mat d = lastDelta.get(i);
            Mat pos = position.get(i);
            delta.add(PoolingLayer.upsample(d, pos));
        }
    }

    public void computeDeltaFromPoolingLayer(List<Mat> lastDelta, List<Mat> lastFilter) {
        for (int i = 0; i < lastDelta.size(); i++) {
            Mat d = lastDelta.get(i);
            Mat f = lastFilter.get(i);
//            Mat paddingDelta = paddingZeor(d, f.rows() - 1, f.cols() - 1, f.rows() - 1,
//                    f.cols() - 1);
            Mat dst = new Mat(d.size(), d.type());
            Imgproc.filter2D(d, dst, d.depth(), f);
            System.out.println(dst);
            delta.add(dst);
        }
    }
    public void updateFilter() {
        for (int i = 0; i < filters.size(); i++) {
            List<Mat> s = filters.get(i);
            Mat d = delta.get(i);
            for (int j = 0; j < s.size(); j++) {
                Mat f = s.get(j);
                Mat m = input.get(j);
                Mat w = new Mat(f.size(), f.type());
                Imgproc.filter2D(m, w, m.depth(), d);
                Core.add(w, f, f);
            }
        }
    }

    public static Mat paddingZeor(Mat mat, int top, int right, int bottom, int left) {
        int srcRows = mat.rows();
        int srcCols = mat.cols();
        Mat result = new Mat(srcRows + top + bottom, srcCols + left + right, mat.type());
        List<Mat> data = new ArrayList<Mat>();
        for (int i = 0; i < top; i++) {
            data.add(Mat.zeros(1, result.cols(), result.type()));
        }
        Mat leftMat;
        Mat rightMat;
        Mat row;
        List<Mat> tmpList = new ArrayList<Mat>();
        for (int i = 0; i < srcRows; i++) {
            tmpList.clear();
            leftMat = Mat.zeros(1, left, result.type());
            rightMat = Mat.zeros(1, right, result.type());
            row = new Mat(1, left + srcCols + right, result.type());
            tmpList.add(leftMat);
            tmpList.add(mat.rowRange(i, i + 1));
            tmpList.add(rightMat);
            Core.hconcat(tmpList, row);
            data.add(row);
        }
        for (int i = 0; i < bottom; i++) {
            data.add(Mat.zeros(1, result.cols(), result.type()));
        }
        Core.vconcat(data, result);
        return result;
    }

    public void print() {
        System.out.println("/***********************************************");
        for (int i = 0; i < filters.size(); i++) {
            System.out.println("filter " + i + ":");
            for (int j = 0; j < filters.get(i).size(); j++) {
                System.out.println(filters.get(i).get(j).dump());
            }
            System.out.println("out " + i + ":");
            System.out.println(out.get(i).dump());
        }
        System.out.println("***********************************************/");
    }

    public List<Mat> getDelta() {
        return delta;
    }

    public void setDelta(List<Mat> delta) {
        this.delta = delta;
    }

    public int getFilterSize() {
        return filterSize;
    }

    public void setFilterSize(int filterSize) {
        this.filterSize = filterSize;
    }

    public int getPadding() {
        return padding;
    }

    public void setPadding(int padding) {
        this.padding = padding;
    }

    public int getStride() {
        return stride;
    }

    public void setStride(int stride) {
        this.stride = stride;
    }

    public int getNumberOfFilters() {
        return numberOfFilters;
    }

    public void setNumberOfFilters(int numberOfFilters) {
        this.numberOfFilters = numberOfFilters;
    }

    public List<Mat> getInput() {
        return input;
    }

    public void setInput(List<Mat> input) {
        this.input = input;
    }

    public List<List<Mat>> getFilters() {
        return filters;
    }

    public void setFilters(List<List<Mat>> filters) {
        this.filters = filters;
    }

    public List<Mat> getOut() {
        return out;
    }

    public void setOut(List<Mat> out) {
        this.out = out;
    }
}
