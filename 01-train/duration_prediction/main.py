import click

from datetime import datetime

from duration_prediction.train import train


@click.command()
@click.option('--train-month', required=True, help='Training month in YYYY-MM format')
@click.option('--val-month', required=True, help='Validation month in YYYY-MM format')
@click.option('--model-output-path', required=True, help='Path where the trained model will be saved')
def main(train_month: str, val_month: str, model_output_path: str):
    train_year, train_month = train_month.split('-')
    train_year = int(train_year)
    train_month = int(train_month)

    val_year, val_month = val_month.split('-')
    val_year = int(val_year)
    val_month = int(val_month)
    
    train_month = datetime(year=train_year, month=train_month, day=1)
    val_month = datetime(year=val_year, month=val_month, day=1)
    
    train(
        train_month=train_month,
        val_month=val_month,
        model_output_path=model_output_path
    )


if __name__ == '__main__':
    main()