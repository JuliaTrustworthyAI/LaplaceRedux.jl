window.BENCHMARK_DATA = {
  "lastUpdate": 1687340661637,
  "repoUrl": "https://github.com/navimakarov/LaplaceRedux.jl",
  "entries": {
    "Julia benchmark result": [
      {
        "commit": {
          "author": {
            "email": "a.ionescu-5@student.tudelft.nl",
            "name": "Andrei Ionescu",
            "username": "Andrei32Ionescu"
          },
          "committer": {
            "email": "a.ionescu-5@student.tudelft.nl",
            "name": "Andrei Ionescu",
            "username": "Andrei32Ionescu"
          },
          "distinct": true,
          "id": "cab3fbd4655a9b207453bfe45e24f0f7e0471b9e",
          "message": "Clean up and add final benchmarks",
          "timestamp": "2023-06-19T18:42:30+02:00",
          "tree_id": "3bab8f7a9d4f9221e11610239d177fc4d59f5193",
          "url": "https://github.com/navimakarov/LaplaceRedux.jl/commit/cab3fbd4655a9b207453bfe45e24f0f7e0471b9e"
        },
        "date": 1687196827814,
        "tool": "julia",
        "benches": [
          {
            "name": "fit_la_batched",
            "value": 329639881,
            "unit": "ns",
            "extra": "gctime=51842074.5\nmemory=1061381648\nallocs=1457837\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "fit_la_unbatched",
            "value": 617895139,
            "unit": "ns",
            "extra": "gctime=97078048\nmemory=2089649168\nallocs=2769052\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "a.ionescu-5@student.tudelft.nl",
            "name": "Andrei Ionescu",
            "username": "Andrei32Ionescu"
          },
          "committer": {
            "email": "a.ionescu-5@student.tudelft.nl",
            "name": "Andrei Ionescu",
            "username": "Andrei32Ionescu"
          },
          "distinct": true,
          "id": "42b2d50d51bda4b81fbadfdcbed8b425b922e5d8",
          "message": "Increase alert limit to 150%",
          "timestamp": "2023-06-19T20:27:36+02:00",
          "tree_id": "415766cd27e128c384fc1fa8f943d6077bb157fd",
          "url": "https://github.com/navimakarov/LaplaceRedux.jl/commit/42b2d50d51bda4b81fbadfdcbed8b425b922e5d8"
        },
        "date": 1687200099701,
        "tool": "julia",
        "benches": [
          {
            "name": "fit_la_batched",
            "value": 344652835,
            "unit": "ns",
            "extra": "gctime=44749696\nmemory=1061381648\nallocs=1457837\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "fit_la_unbatched",
            "value": 662740173,
            "unit": "ns",
            "extra": "gctime=84151402\nmemory=2089649168\nallocs=2769052\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "a.ionescu-5@student.tudelft.nl",
            "name": "Andrei Ionescu",
            "username": "Andrei32Ionescu"
          },
          "committer": {
            "email": "a.ionescu-5@student.tudelft.nl",
            "name": "Andrei Ionescu",
            "username": "Andrei32Ionescu"
          },
          "distinct": true,
          "id": "dffbfa119f3773b59acc522fc09b1ba44e8b2a5e",
          "message": "Re-add removed tests to workflow",
          "timestamp": "2023-06-19T21:29:15+02:00",
          "tree_id": "713c7ab14c7d10d549b0e3f11f582bacb75f8fb5",
          "url": "https://github.com/navimakarov/LaplaceRedux.jl/commit/dffbfa119f3773b59acc522fc09b1ba44e8b2a5e"
        },
        "date": 1687204418324,
        "tool": "julia",
        "benches": [
          {
            "name": "fit_la_batched",
            "value": 333733956,
            "unit": "ns",
            "extra": "gctime=49891833\nmemory=1061381648\nallocs=1457837\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "fit_la_unbatched",
            "value": 636841907.5,
            "unit": "ns",
            "extra": "gctime=95285456.5\nmemory=2089649168\nallocs=2769052\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "89662728+Andrei32Ionescu@users.noreply.github.com",
            "name": "Andrei32Ionescu",
            "username": "Andrei32Ionescu"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b39a99e7fcf2a108fb91259635b92c95a64020e9",
          "message": "Merge pull request #33 from navimakarov/pipeline-improvements\n\nPipeline improvements",
          "timestamp": "2023-06-20T17:50:13+02:00",
          "tree_id": "713c7ab14c7d10d549b0e3f11f582bacb75f8fb5",
          "url": "https://github.com/navimakarov/LaplaceRedux.jl/commit/b39a99e7fcf2a108fb91259635b92c95a64020e9"
        },
        "date": 1687277851178,
        "tool": "julia",
        "benches": [
          {
            "name": "fit_la_batched",
            "value": 378245082,
            "unit": "ns",
            "extra": "gctime=56327628\nmemory=1061381648\nallocs=1457837\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "fit_la_unbatched",
            "value": 705931333,
            "unit": "ns",
            "extra": "gctime=106743259.5\nmemory=2089649168\nallocs=2769052\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "69299116+severinbratus@users.noreply.github.com",
            "name": "severinbratus",
            "username": "severinbratus"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3f64270ba36c56e540323d3fdba6bb92842c6178",
          "message": "Merge pull request #32 from navimakarov/6-block-diagonal-hessian-approximations\n\nBlock-diagonal Hessian approximations: KFAC Fisher on multi-class",
          "timestamp": "2023-06-20T18:07:32+02:00",
          "tree_id": "bb868d56e186cdb1fddee5be4a60406e2f6ac7e1",
          "url": "https://github.com/navimakarov/LaplaceRedux.jl/commit/3f64270ba36c56e540323d3fdba6bb92842c6178"
        },
        "date": 1687279297349,
        "tool": "julia",
        "benches": [
          {
            "name": "fit_la_batched",
            "value": 346417504,
            "unit": "ns",
            "extra": "gctime=57025974\nmemory=1061381648\nallocs=1457837\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "fit_la_unbatched",
            "value": 637357389,
            "unit": "ns",
            "extra": "gctime=97098322\nmemory=2089649168\nallocs=2769052\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "navimakarov",
            "username": "navimakarov"
          },
          "committer": {
            "name": "navimakarov",
            "username": "navimakarov"
          },
          "id": "3f64270ba36c56e540323d3fdba6bb92842c6178",
          "message": "Final merge",
          "timestamp": "2023-05-31T20:38:56Z",
          "url": "https://github.com/navimakarov/LaplaceRedux.jl/pull/35/commits/3f64270ba36c56e540323d3fdba6bb92842c6178"
        },
        "date": 1687283398576,
        "tool": "julia",
        "benches": [
          {
            "name": "fit_la_batched",
            "value": 424470318.5,
            "unit": "ns",
            "extra": "gctime=61363441\nmemory=1061381648\nallocs=1457837\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "fit_la_unbatched",
            "value": 806752951,
            "unit": "ns",
            "extra": "gctime=116527544\nmemory=2089649168\nallocs=2769052\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "94811276+MarkArdman@users.noreply.github.com",
            "name": "MarkArdman",
            "username": "MarkArdman"
          },
          "committer": {
            "email": "94811276+MarkArdman@users.noreply.github.com",
            "name": "MarkArdman",
            "username": "MarkArdman"
          },
          "distinct": true,
          "id": "afc1702ec44cd720d7f2475efdec8389122e723f",
          "message": "Resolve instead of instantiate",
          "timestamp": "2023-06-21T01:30:36+02:00",
          "tree_id": "2c685a220001dac42b4d77497598403043d967d6",
          "url": "https://github.com/navimakarov/LaplaceRedux.jl/commit/afc1702ec44cd720d7f2475efdec8389122e723f"
        },
        "date": 1687305295210,
        "tool": "julia",
        "benches": [
          {
            "name": "fit_la_batched",
            "value": 412213509,
            "unit": "ns",
            "extra": "gctime=54609564\nmemory=1061381648\nallocs=1457837\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "fit_la_unbatched",
            "value": 772111848,
            "unit": "ns",
            "extra": "gctime=107005179\nmemory=2089649168\nallocs=2769052\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "94811276+MarkArdman@users.noreply.github.com",
            "name": "MarkArdman",
            "username": "MarkArdman"
          },
          "committer": {
            "email": "94811276+MarkArdman@users.noreply.github.com",
            "name": "MarkArdman",
            "username": "MarkArdman"
          },
          "distinct": true,
          "id": "17a28bba28b119e7a876204771e58213c823d620",
          "message": "Restore the pipeline",
          "timestamp": "2023-06-21T01:41:49+02:00",
          "tree_id": "4df7815baf2de0541ba7ea273dce3d474c292d5f",
          "url": "https://github.com/navimakarov/LaplaceRedux.jl/commit/17a28bba28b119e7a876204771e58213c823d620"
        },
        "date": 1687306693984,
        "tool": "julia",
        "benches": [
          {
            "name": "fit_la_batched",
            "value": 364157519,
            "unit": "ns",
            "extra": "gctime=48313802\nmemory=1061381648\nallocs=1457837\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "fit_la_unbatched",
            "value": 676038165,
            "unit": "ns",
            "extra": "gctime=90360660\nmemory=2089649168\nallocs=2769052\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "94811276+MarkArdman@users.noreply.github.com",
            "name": "Mark Ardman",
            "username": "MarkArdman"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "2f59450891e9cc7e4d12e701bb95260245416661",
          "message": "Merge pull request #36 from navimakarov/10-mlj-interfacing\n\nMlj interfacing",
          "timestamp": "2023-06-21T11:22:26+02:00",
          "tree_id": "4df7815baf2de0541ba7ea273dce3d474c292d5f",
          "url": "https://github.com/navimakarov/LaplaceRedux.jl/commit/2f59450891e9cc7e4d12e701bb95260245416661"
        },
        "date": 1687340597847,
        "tool": "julia",
        "benches": [
          {
            "name": "fit_la_batched",
            "value": 312495516.5,
            "unit": "ns",
            "extra": "gctime=41178074\nmemory=1061381648\nallocs=1457837\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "fit_la_unbatched",
            "value": 629645503,
            "unit": "ns",
            "extra": "gctime=83803040.5\nmemory=2089649168\nallocs=2769052\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "navimakarov",
            "username": "navimakarov"
          },
          "committer": {
            "name": "navimakarov",
            "username": "navimakarov"
          },
          "id": "2f59450891e9cc7e4d12e701bb95260245416661",
          "message": "Final merge",
          "timestamp": "2023-05-31T20:38:56Z",
          "url": "https://github.com/navimakarov/LaplaceRedux.jl/pull/35/commits/2f59450891e9cc7e4d12e701bb95260245416661"
        },
        "date": 1687340657557,
        "tool": "julia",
        "benches": [
          {
            "name": "fit_la_batched",
            "value": 308407330.5,
            "unit": "ns",
            "extra": "gctime=39279399\nmemory=1061381648\nallocs=1457837\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          },
          {
            "name": "fit_la_unbatched",
            "value": 596717437,
            "unit": "ns",
            "extra": "gctime=76736586\nmemory=2089649168\nallocs=2769052\nparams={\"gctrial\":true,\"time_tolerance\":0.05,\"samples\":10000,\"evals\":1,\"gcsample\":false,\"seconds\":5,\"overhead\":0,\"memory_tolerance\":0.01}"
          }
        ]
      }
    ]
  }
}