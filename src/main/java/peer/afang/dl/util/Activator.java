package peer.afang.dl.util;

import org.opencv.core.Mat;

/**
 * @author ZhangZhenfang
 * @date 2019/2/3 9:30
 */
public interface Activator {
    Mat activate(Mat mat);
    Mat derivative(Mat mat);
    Mat activate(Mat src, Mat dst);
    Mat derivative(Mat src, Mat dst);
    double activate(double x);
    double derivative(double x);
}
