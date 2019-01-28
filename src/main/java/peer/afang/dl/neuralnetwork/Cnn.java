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
        MnistReader reader = new MnistReader(MnistReader.TRAIN_IMAGES_FILE, MnistReader.TRAIN_LABELS_FILE);
        double[] nextImage = reader.getNextImage(false);
        Mat m = new Mat(28, 28, CvType.CV_32F);
        m.put(0, 0, nextImage);
        m.reshape(28);
        List<Mat> input = new ArrayList();
        input.add(m);

        ConvLayer convLayer1 = new ConvLayer(input, 3, 1, 1, 1);
        convLayer1.computeOut();
        convLayer1.print();

        PoolingLayer poolingLayer1 = new PoolingLayer(convLayer1.getOut(), 2, 0, 2, 2);
        poolingLayer1.computeOut();
        poolingLayer1.print();
        List<Mat> out = poolingLayer1.getOut();

        ConvLayer convLayer2 = new ConvLayer(out, 3, 1, 1, 3);
        convLayer2.computeOut();
        convLayer2.print();

        PoolingLayer poolingLayer2 = new PoolingLayer(convLayer2.getOut(), 2, 0, 2, 1);
        poolingLayer2.computeOut();
        poolingLayer2.print();

        List<Mat> out1 = poolingLayer2.getOut();
        Mat bpInput = new Mat(out1.size(), 49, CvType.CV_32F);
        List<Mat> mats = new ArrayList<Mat>();

        for (int i = 0; i < out1.size(); i++) {
            mats.add(out1.get(i).reshape(1, 1));
        }
        Core.vconcat(mats, bpInput);
        bpInput = bpInput.reshape(1, 1);
        Bp bp = new Bp(bpInput.cols(), 2, new int[]{20, 15}, 10);
        bp.newForward(bpInput);
        System.out.println(bp.getOutputLayer().dump());
        double[] nextLabel = reader.getNextLabel();
        Mat label = new Mat(10, 1, CvType.CV_32F);
        label.put(0, 0, nextLabel);
        bp.newBackPropagation(bpInput, label, 0.1);
        Mat delta = bp.getDelta();
        System.out.println(delta.dump());
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