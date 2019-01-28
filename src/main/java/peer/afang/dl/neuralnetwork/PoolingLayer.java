package peer.afang.dl.neuralnetwork;

import org.opencv.core.Mat;

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
        int poolSize = (input.get(0).rows() + 2 * padding - filterSize) / stride + 1;
        for (int i = 0; i < input.size(); i++) {
            out.add(new Mat(poolSize, poolSize, input.get(0).type()));
        }
    }

    public void computeOut() {
        try {
            for (int i = 0; i < input.size(); i++) {
                Cnn.maxPooling(input.get(i), filterSize, filterSize, out.get(i));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
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