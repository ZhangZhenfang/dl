package peer.afang.dl.neuralnetwork;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import peer.afang.dl.util.MnistReader;

import java.util.Random;

/**
 * @author ZhangZhenfang
 * @date 19/1/23 14:43
 */
public class Cnn {
    static {
        String path = "/usr/local/share/OpenCV/java/libopencv_java341.so";
        System.load(path);
    }
    private Mat weight;

    public static void main(String[] args) throws Exception {
        MnistReader reader = new MnistReader(MnistReader.TRAIN_IMAGES_FILE, MnistReader.TRAIN_LABELS_FILE);
        double[] nextImage = reader.getNextImage(false);
        Mat m = new Mat(28, 28, CvType.CV_32F);
        m.put(0, 0, nextImage);
        m.reshape(28);
        Cnn cnn = new Cnn();
        cnn.print();
        cnn.forward(m);
    }

    public Cnn() {
        this.weight = new Mat(3, 3, CvType.CV_32F);
        double[] data = new double[weight.cols() * weight.rows()];
        for (int i = 0; i < data.length; i++) {
            data[i] = (new Random().nextDouble() - 0.5) * 2;
        }
        weight.put(0, 0, data);
    }

    public void forward(Mat mat) {
        Mat dst = new Mat(mat.size(), CvType.CV_32F);
        Imgproc.filter2D(mat, dst, mat.depth(), weight);
        System.out.println(dst.dump());
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