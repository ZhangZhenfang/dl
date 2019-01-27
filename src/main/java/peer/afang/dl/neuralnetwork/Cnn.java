package peer.afang.dl.neuralnetwork;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import peer.afang.dl.util.MnistReader;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * @author ZhangZhenfang
 * @date 19/1/23 14:43
 */
public class Cnn {

    static {
        String path = "D:\\openCV\\opencv\\build\\java\\x64\\opencv_java341.dll";
        System.load(path);
    }
    private Mat weight;

    public static void main(String[] args) throws Exception {
//        MnistReader reader = new MnistReader(MnistReader.TRAIN_IMAGES_FILE, MnistReader.TRAIN_LABELS_FILE);
//        double[] nextImage = reader.getNextImage(false);
//        Mat m = new Mat(28, 28, CvType.CV_32F);
//        m.put(0, 0, nextImage);
//        m.reshape(28);

        List input = new ArrayList();
        Mat m = new Mat(5, 5, CvType.CV_32F);
        m.put(0, 0, new double[]{0, 1, 1, 0, 2, 2, 2, 2, 2, 1, 1, 0, 0, 2, 0, 0, 1, 1, 0, 0, 1, 2, 0, 0, 2});
        System.out.println(m.dump());
        input.add(m);
        m = new Mat(5, 5, CvType.CV_32F);
        m.put(0, 0, new double[]{1, 0, 2, 2, 0, 0, 0, 0, 2, 0, 1, 2, 1, 2, 1, 1, 0, 0, 0, 0, 1, 2, 1, 1, 1});
        System.out.println(m.dump());
        input.add(m);
        m = new Mat(5, 5, CvType.CV_32F);
        m.put(0, 0, new double[]{2, 1, 2, 0, 0, 1, 0, 0, 1, 0, 0, 2, 1, 0, 1, 0, 1, 2, 2, 2, 2, 1, 0, 0, 1});
        System.out.println(m.dump());
        input.add(m);
        ConvLayer convLayer = new ConvLayer(input, 3, 1, 1, 2);
        convLayer.computeOut();
        convLayer.print();
//        Core.chan
//        Cnn cnn = new Cnn();
//        cnn.print();
//        cnn.forward(m);
    }

    public Cnn() {
        this.weight = new Mat(3, 3, CvType.CV_32F);
        double[] data = new double[weight.cols() * weight.rows()];
        for (int i = 0; i < data.length; i++) {
            data[i] = (new Random().nextDouble() - 0.5) * 2;
        }
        weight.put(0, 0, data);
    }

    public void forward(Mat mat) throws Exception{
        Mat conDst = new Mat(mat.size(), CvType.CV_32F);
        Imgproc.filter2D(mat, conDst, mat.depth(), weight);
        Mat poolDst = new Mat(conDst.rows() / 2, conDst.cols() / 2, CvType.CV_32F);
        maxPooling(conDst, 2, 2, poolDst);
        System.out.println(conDst.dump());
        System.out.println(poolDst.dump());
    }

    public void print() {
        System.out.println("weight:");
        System.out.println(weight.dump());
    }

    public static void maxPooling(Mat src, int rows, int cols, Mat dst) throws Exception {
        double[] data = new double[dst.cols() * dst.rows()];
        int i = 0;
        if (src.rows() % rows != 0 || src.cols() % cols != 0) {
            throw new Exception("size don't fit");
        }
        if (src.rows() / rows != dst.rows() || src.cols() / cols != dst.cols()) {
            throw new Exception("dst size don't fit");
        }
        for (int r = 0; r < src.rows(); r += rows) {
            for (int c = 0; c < src.cols();  c += cols) {
                Mat mat = src.rowRange(r, r + rows).colRange(c, c + cols);
                Core.MinMaxLocResult minMaxLocResult = Core.minMaxLoc(mat);
                data[i++] = minMaxLocResult.maxVal;
            }
        }
        dst.put(0, 0, data);
    }
}

class ConvLayer {

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
                System.out.println(i + " " + j);
                System.out.println(dst.dump());
                Core.add(dst, out.get(i), out.get(i));
            }
        }
    }

    public void backPropagation() {
        System.out.println("this is backPropagation");
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
}

class PoolingLayer {
    /**
     * 过滤器大小
     */
    private int filterSize;
    /**
     * 步长
     */
    private int stride;
    /**
     *
     */
    private int numberOfFilters;

}