package peer.afang.dl.util;

import org.opencv.core.Mat;

/**
 * @author ZhangZhenfang
 * @date 2019/2/3 9:35
 */
public class Log implements Activator {

    public Mat activate(Mat src, Mat dst) {
        for (int i = 0; i < src.rows(); i++) {
            for (int j = 0; j < src.cols(); j++) {
                dst.put(i, j, activate(src.get(i, j)[0]));
            }
        }
        return null;
    }

    public Mat derivative(Mat src, Mat dst) {
        for (int i = 0; i < src.rows(); i++) {
            for (int j = 0; j < src.cols(); j++) {
                dst.put(i, j, derivative(src.get(i, j)[0]));
            }
        }
        return null;
    }
    public Mat activate(Mat mat) {
        Mat result = new Mat(mat.size(), mat.type());
        for (int i = 0; i < mat.rows(); i++) {
            for (int j = 0; j < mat.cols(); j++) {
                result.put(i, j, activate(mat.get(i, j)[0]));
            }
        }
        return result;
    }
    public Mat derivative(Mat mat) {
        Mat result = new Mat(mat.size(), mat.type());
        for (int i = 0; i < mat.rows(); i++) {
            for (int j = 0; j < mat.cols(); j++) {
                result.put(i, j, derivative(mat.get(i, j)[0]));
            }
        }
        return result;
    }
    public double activate(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public double derivative(double x) {
        double t = activate(x);
        return t * (1 - t);
    }
}
