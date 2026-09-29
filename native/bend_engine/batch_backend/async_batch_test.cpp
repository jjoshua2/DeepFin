// Bounded native-worker ownership tests, no model or Bend tree here.
#include "async_batch.h"
#include <atomic>
#include <chrono>
#include <future>
#include <iostream>

using deepfin_native::AsyncBatch;
using namespace std::chrono_literals;
static unsigned checks = 0;
static void require(bool yes) { ++checks; if (!yes) throw std::runtime_error("test assertion failed"); }
template<class F> void rejected(F action) { bool bad=false; try { action(); } catch (const std::exception&) {bad=true;} require(bad); }
struct Gate {
  std::mutex mutex; std::condition_variable cv; bool started=false, released=false;
  void block() {std::unique_lock l(mutex); started=true; cv.notify_all(); cv.wait(l,[&]{return released;});}
  void wait() {std::unique_lock l(mutex); require(cv.wait_for(l,5s,[&]{return started;}));}
  void release() {std::lock_guard l(mutex); released=true; cv.notify_all();}
};
static AsyncBatch::Status finish(AsyncBatch& worker,uint32_t token,float* out,uint32_t rows) {
  const auto deadline=std::chrono::steady_clock::now()+5s;
  for (;;) {auto s=worker.take(token,out,rows); if(s!=AsyncBatch::pending) return s;
    if(std::chrono::steady_clock::now()>deadline) throw std::runtime_error("test timeout");
    std::this_thread::yield();}
}
int main() {
  for(uint32_t batch: {1u,2u,4u,8u,16u}) for(uint32_t channels: {146u,175u}) {
    Gate gate; unsigned calls=0; const float* first_in=nullptr; float* first_out=nullptr;
    std::vector<float> input(size_t(batch)*channels*64,1), output(size_t(batch)*1861+1,-19);
    AsyncBatch w(batch,channels,[&](const float* x,uint32_t rows,float* y,uint32_t count){
      ++calls; if(calls==1) {first_in=x;first_out=y;gate.block();}
      if(x!=first_in||y!=first_out||count!=rows*1861) return 1;
      for(uint32_t r=0;r<rows;++r) for(uint32_t j=0;j<1861;++j) y[r*1861+j]=x[size_t(r)*channels*64]+float(j);
      return 0;
    });
    const auto token=w.submit(input.data(),batch); require(token==1); gate.wait();
    require(w.submit(input.data(),batch)==0);
    std::fill(input.begin(),input.end(),999); // In-flight snapshot must remain 1.
    require(w.take(token+1,output.data(),batch)==AsyncBatch::unknown);
    rejected([&]{w.take(token,nullptr,batch);}); rejected([&]{w.take(token,output.data(),0);});
    require(w.take(token,output.data(),batch)==AsyncBatch::pending); require(output[0]==-19);
    gate.release(); require(finish(w,token,output.data(),batch)==AsyncBatch::complete);
    for(uint32_t r=0;r<batch;++r) require(output[r*1861]==1);
    require(output.back()==-19); require(w.take(token,output.data(),batch)==AsyncBatch::unknown);
    for(uint32_t i=0;i<16;++i) {
      uint32_t rows=1+i%batch;std::fill(input.begin(),input.end(),float(i+3));std::fill(output.begin(),output.end(),-19);
      auto next=w.submit(input.data(),rows);require(next==i+2);require(finish(w,next,output.data(),rows)==AsyncBatch::complete);
      for(uint32_t r=0;r<rows;++r) require(output[r*1861+7]==float(i+10));
      require(output[size_t(rows)*1861]==-19);require(w.take(token,output.data(),batch)==AsyncBatch::unknown);
    }
    rejected([&]{w.submit(nullptr,1);}); rejected([&]{w.submit(input.data(),0);}); rejected([&]{w.submit(input.data(),batch+1);});
    w.shutdown();w.shutdown();rejected([&]{w.submit(input.data(),1);});require(calls==17);
  }
  for(bool throws: {false,true}) {
    std::vector<float> x(146*64,1), y(1861,-27);
    AsyncBatch w(1,146,[&](const float*,uint32_t,float* out,uint32_t){out[0]=123;
      if(throws) throw std::runtime_error("injected");return 1;});
    auto t=w.submit(x.data(),1);require(finish(w,t,y.data(),1)==AsyncBatch::failed);require(y[0]==-27);
    require(w.take(t,y.data(),1)==AsyncBatch::unknown);rejected([&]{w.submit(x.data(),1);});
  }
  {Gate gate;std::vector<float> x(146*64,1);std::atomic<bool> done=false;
    AsyncBatch w(1,146,[&](const float*,uint32_t,float*,uint32_t){gate.block();done=true;return 0;});
    w.submit(x.data(),1);gate.wait();auto wait=std::async(std::launch::async,[&]{w.shutdown();});
    require(wait.wait_for(20ms)==std::future_status::timeout);gate.release();wait.get();require(done);
  }
  {std::vector<float>x(146*64,1),y(1861); AsyncBatch w(1,146,[](const float*,uint32_t,float*,uint32_t){return 0;},0xfffffffe);
    auto t=w.submit(x.data(),1);require(t==0xffffffff);require(finish(w,t,y.data(),1)==AsyncBatch::complete);
    rejected([&]{w.submit(x.data(),1);});}
  for(uint32_t n:{0u,3u,17u}) rejected([&]{AsyncBatch w(n,146,{});});
  rejected([]{AsyncBatch w(1,174,{});});
  std::cout<<"{\"status\":\"passed\",\"assertions\":"<<checks<<",\"configurations\":10,\"reused_calls\":160}\n";
}
