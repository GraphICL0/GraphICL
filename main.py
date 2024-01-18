import time
import arg
import os

os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"


def main():
    parser = arg.get_parser()
    args = parser.parse_args()
    print(args)

    if args.embedder == 'graphicl':
        from models.graphicl import graphicl
        embedder = graphicl(args)

    t_total = time.time()
    embedder.training()
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))


if __name__ == '__main__':
    main()
