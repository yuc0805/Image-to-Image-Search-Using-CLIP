import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('Image-to-Image Search Engine',add_help=False)
    
    
    # Dataset parameters
    parser.add_argument('--data_path',default='./Fruit-Vegetables-Images',
                        help='path of the datasets')


def main(args):

    return

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)