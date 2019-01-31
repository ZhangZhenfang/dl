package peer.afang.dl.neuralnetwork.cnn;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import peer.afang.dl.util.MatUtils;

import java.util.ArrayList;
import java.util.List;

/**
 * @author ZhangZhenfang
 * @date 2019/1/28 9:58
 */
public class PoolingLayer {
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
    private List<Mat> out;
    private List<Mat> position;
    private List<Mat> delta;

    double[][][] testdata = new double[][][]{{{-1, 1, 0, 0, 1, 0, 0, 1, 1}, {-1, -1, 0, 0, 0, 0, 0, -1, 0}, {0, 0, -1, 0, 1, 0, 1, -1 , -1}}, {{1, 1, -1, -1, -1, 1, 0, -1, 1}, {0, 1, 0, -1, 0, -1, -1, 1, 0}, {-1, 0, 0, -1, 0, 1, -1, 0, 0}}};
    public PoolingLayer() {

    }
    public PoolingLayer(List<Mat> input, int filterSize, int padding, int stride, int numberOfFilters) {
        this.input = input;
        this.filterSize = filterSize;
        this.padding = padding;
        this.stride = stride;
        this.numberOfFilters = numberOfFilters;
        init();
    }

    private void init() {
        out = new ArrayList<Mat>();
        position = new ArrayList<Mat>();
        delta = new ArrayList<Mat>();
        int poolSize = (input.get(0).rows() + 2 * padding - filterSize) / stride + 1;
        for (int i = 0; i < input.size(); i++) {
            out.add(new Mat(poolSize, poolSize, input.get(0).type()));
            position.add(new Mat(poolSize, poolSize, CvType.CV_16U));
        }
    }

    public void computeOut() {
        try {
            for (int i = 0; i < input.size(); i++) {
                Cnn.maxPooling(input.get(i), filterSize, filterSize, out.get(i), position.get(i));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void computeDeltaFromConvLayer(List<Mat> lastDelta, List<Mat> lastFilter) {
        delta.clear();
        for (int i = 0; i < lastDelta.size(); i++) {
            Mat d = lastDelta.get(i);
            Mat f = lastFilter.get(i);
            Mat tmp = new Mat(f.size(), f.type());
            Core.flip(f, tmp, 0);
            Core.flip(tmp, tmp, 1);
//            Mat paddingDelta = ConvLayer.paddingZeor(d, f.rows() - 1, f.cols() - 1, f.rows() - 1,
//                    f.cols() - 1);
            Mat dst = new Mat(d.size(), d.type());
            int paddingSize = lastFilter.get(0).rows() - 1;
            Mat mat = MatUtils.paddingZeor(d, paddingSize, paddingSize, paddingSize, paddingSize);
            Mat conv = MatUtils.conv(mat, tmp, 1, 0, 0);
            delta.add(conv);
        }
    }

    public static Mat upsample(Mat mat, Mat pos) {
        List<Mat> list1 = new ArrayList<Mat>();
        for (int i = 0; i < pos.rows(); i++) {
            List<Mat> list2 = new ArrayList<Mat>();
            for (int j = 0; j < pos.cols(); j++) {
                double[] data = new double[4];
                data[(int) pos.get(i, j)[0]] = mat.get(i, j)[0];
                Mat m = new Mat(1, 4, CvType.CV_32F);
                m.put(0, 0, data);
                m = m.reshape(1, 2);
                list2.add(m);
            }
            Mat dst = new Mat(2, pos.cols() * 2, CvType.CV_32F);
            Core.hconcat(list2, dst);
            list1.add(dst);
        }
        Mat result = new Mat(pos.rows() * 2, pos.cols() * 2, CvType.CV_32F);
        Core.vconcat(list1, result);
        return result;
    }
    public void backPropagation() {
        System.out.println("this is backPropagation");
    }

    public void print() {
        System.out.println("/***********************************************");
        for (int i = 0; i < out.size(); i++) {
            System.out.println("input " + i);
            System.out.println(input.get(i).dump());
            System.out.println("out " + i);
            System.out.println(out.get(i).dump());
        }
        System.out.println("***********************************************/");
    }

    public List<Mat> getPosition() {
        return position;
    }

    public void setPosition(List<Mat> position) {
        this.position = position;
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

    public List<Mat> getOut() {
        return out;
    }

    public void setOut(List<Mat> out) {
        this.out = out;
    }
}