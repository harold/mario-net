(ns mario-net.one-d
  (:require [clojure.core.matrix :as m]
            [cortex.description :as desc]
            [cortex.network :as net]))

(def network-desc
  [(desc/input 1)
   (desc/linear->relu 32)
   (desc/linear->relu 32)
   (desc/linear->relu 32)
   (desc/linear->relu 32)
   (desc/linear 1)])

(defn square
  [x]
  (* x x))

(defn build-dataset
  "Evenly spreads n points over [lower upper) and
  makes pairs [x (f x)]."
  [f lower upper n]
  (for [i (range n)]
    (let [x (+ lower (* (/ i n) (- upper lower)))]
      [[(double x)] (double (f x))])))

(defn train-network
  []
  (let [dataset (build-dataset square 0 10 1000)
        training-data (mapv first dataset)
        training-labels (mapv second dataset)
        network (desc/build-and-create-network network-desc)]
    (println "Training network:" (count training-data) "inputs...")
    (net/train-until-error-stabilizes network training-data training-labels)))

(defn run
  [net lower upper n]
  (->> (for [i (range n)]
         [(double (+ lower (* (/ i n) (- upper lower))))])
       (net/run net)
       (map (comp first seq))))

(comment
  (def best-net (train-network))
  (run best-net -20 20 100)
  )
