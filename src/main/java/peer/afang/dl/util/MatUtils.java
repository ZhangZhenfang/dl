package peer.afang.dl.util;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.List;

/**
 * @author ZhangZhenfang
 * @date 2019/1/31 10:35
 */
public class MatUtils {
    static {
        String path = "D:\\openCV\\opencv\\build\\java\\x64\\opencv_java341.dll";
        System.load(path);
    }

    public static void main(String[] args) {
        Mat src = new Mat(1, 16, CvType.CV_32F);
        src.put(0, 0, new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
        Mat kernel = Mat.ones(1, 4, CvType.CV_32F);
        Mat dst = new Mat(1, 13, CvType.CV_32F);
        conv(src, kernel, dst, 1, 0, 0);
        System.out.println(src.dump());
        System.out.println(dst.dump());

    }

    public static Mat conv(Mat src, Mat kernel, int stride, int padding, double d) {
        int filterRows = kernel.rows();
        int filterCols = kernel.cols();

        int resultRows = (src.rows() + 2 * padding - filterRows) / stride + 1;
        int resultCols = (src.cols() + 2 * padding - filterCols) / stride + 1;

//        System.out.println(resultRows);
//        System.out.println(resultCols);
        Mat result = new Mat(resultRows, resultCols, CvType.CV_32F);
        double[] data = new double[resultRows * resultCols];
        int index = 0;
        for (int i = 0; i <= src.rows() - filterRows; i++) {
            for (int j = 0; j <= src.cols() - filterCols; j++) {
                Mat submat = src.submat(i, i + filterRows, j, j + filterCols);
                Mat res = new Mat();
                Core.multiply(submat, kernel, res);
                data[index++] = sumMat(res);
            }
        }
        result.put(0, 0, data);
        return result;
    }
    public static void conv(Mat src, Mat kernel, Mat dst, int stride, int padding, double d) {
        int filterRows = kernel.rows();
        int filterCols = kernel.cols();

        int resultRows = (src.rows() + 2 * padding - filterRows) / stride + 1;
        int resultCols = (src.cols() + 2 * padding - filterCols) / stride + 1;

        double[] data = new double[resultRows * resultCols];
        int index = 0;
        for (int i = 0; i <= src.rows() - filterRows; i++) {
            for (int j = 0; j <= src.cols() - filterCols; j++) {
                Mat submat = src.submat(i, i + filterRows, j, j + filterCols);
                Mat res = new Mat();
                Core.multiply(submat, kernel, res);
                data[index++] = sumMat(res);
            }
        }
        dst.put(0, 0, data);
    }

    public static double sumMat(Mat mat) {
        double result = 0;
        for (int i = 0; i < mat.rows(); i++) {
            for (int j = 0; j < mat.cols(); j++) {
                result += mat.get(i, j)[0];
            }
        }
        return result;
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
}
