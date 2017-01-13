// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <vector>

#include "caffe/layers/dropproject_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void DropprojectLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  threshold_ = this->layer_param_.dropproject_param().dropproject_ratio();
  DCHECK(threshold_ > 0.);
  DCHECK(threshold_ < 1.);
  scale_ = 1. / (1. - threshold_);
  uint_thres_ = static_cast<unsigned int>(UINT_MAX * threshold_);
}

template <typename Dtype>
void DropprojectLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  const int count = bottom[0]->count();
  // Set up the cache for random number generation
  // ReshapeLike does not work because rand_vec_ is of Dtype uint
  rand_vec_.Reshape(bottom[0]->shape());
  // Rotation vector is square of number of number of parameters in bottom
  rand_rot_.Reshape(1, 1, count, count);
  caffe_rng_uniform(count*count, (Dtype) 0., (Dtype) 1., rand_rot_.mutable_cpu_data());
  // Calculate rotation matrix using QR factorization
  vector<Dtype> rwork(count*count);
//#ifdef CPU_ONLY
  caffe_cpu_linalg_qr(count, count, rand_rot_.mutable_cpu_data(), rwork.data());
//#else
//  caffe_gpu_linalg_qr(count, count, rand_rot_.mutable_gpu_data(), rwork.data());
//#endif
  // TODO 
  // if numpy.linalg.det(self.rotation) < 0:
  //              self.rotation[:, 0] = -self.rotation[:, 0]
}

template <typename Dtype>
void DropprojectLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  unsigned int* mask = rand_vec_.mutable_cpu_data();
  const Dtype* rotation = rand_rot_.cpu_data();
  const int count = bottom[0]->count();

  // Rotate parameter space
  // TODO xp.dot(self.rotation.T, xp.dot(self.rotation, x[0].T) * self.mask.T).T,
  /*caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
      M_, N_, K_, (Dtype)1.,
      bottom_data, rotation, (Dtype)0., top_data);*/

  if (this->phase_ == TRAIN) {
    // Create random numbers
    caffe_rng_bernoulli(count, 1. - threshold_, mask);
    for (int i = 0; i < count; ++i) {
      top_data[i] = top_data[i] * mask[i] * scale_;
    }
  } else {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }
  // Rotate parameter space back
  // return xp.dot(self.rotation.T, xp.dot(self.rotation, gy[0].T) * self.mask.T).T,
  /*caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
      M_, N_, K_, (Dtype)1.,
      bottom_data, rotation, (Dtype)0., top_data);*/
}

template <typename Dtype>
void DropprojectLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    if (this->phase_ == TRAIN) {
      const unsigned int* mask = rand_vec_.cpu_data();
      const int count = bottom[0]->count();
      for (int i = 0; i < count; ++i) {
        bottom_diff[i] = top_diff[i] * mask[i] * scale_;
      }
    } else {
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(DropprojectLayer);
#endif

INSTANTIATE_CLASS(DropprojectLayer);
REGISTER_LAYER_CLASS(Dropproject);

}  // namespace caffe