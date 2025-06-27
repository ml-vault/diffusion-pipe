import json
import os
from collections import deque
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import logging

import torch
from deepspeed.utils.logging import logger as ds_logger
from utils.common import is_main_process

class LossDebugger:
    """
    I2V LoRAトレーニング用のLossデバッグユーティリティ
    
    機能:
    - ファイル単位のloss追跡
    - 閾値ベースアラート
    - JSON出力機能
    - TensorBoard統合
    - メモリ効率を考慮した実装（deque使用）
    - 分散学習環境対応
    """
    
    def __init__(
        self,
        enabled: bool = True,
        loss_threshold: float = 10.0,
        max_history: int = 1000,
        output_dir: Optional[str] = None,
        alert_frequency: int = 10,
        save_frequency: int = 100
    ):
        """
        Args:
            enabled: デバッグ機能の有効/無効
            loss_threshold: アラートを発生させるloss閾値
            max_history: メモリに保持する最大履歴数
            output_dir: JSONファイル出力ディレクトリ
            alert_frequency: アラートの最大頻度（ステップ数）
            save_frequency: JSONファイル保存頻度（ステップ数）
        """
        self.enabled = enabled and is_main_process()
        if not self.enabled:
            return
            
        self.loss_threshold = loss_threshold
        self.max_history = max_history
        self.alert_frequency = alert_frequency
        self.save_frequency = save_frequency
        
        # 履歴データ（メモリ効率のためdeque使用）
        self.loss_history = deque(maxlen=max_history)
        self.file_loss_stats = {}  # ファイル別統計
        
        # アラート制御
        self.last_alert_step = -alert_frequency
        self.step_count = 0
        
        # 出力設定
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.json_output_path = self.output_dir / "loss_debug.json"
        
        # ログ設定
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[LossDebugger] %(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def log_loss(
        self,
        loss: float,
        step: int,
        batch_data: Union[Dict[str, Any], tuple, Any],
        tb_writer: Optional[Any] = None,
        current_files: Optional[List[str]] = None
    ) -> None:
        """
        Lossをログに記録
        
        Args:
            loss: 計算されたloss値
            step: 現在のトレーニングステップ
            batch_data: バッチデータ（辞書形式またはタプル形式）
            tb_writer: TensorBoardライター（オプション）
            current_files: 現在のバッチのファイル情報（オプション）
        """
        if not self.enabled:
            return
        
        try:
            self.step_count = step
            
            # ファイル名を取得（複数の方法を試行）
            image_files = []
            
            # 1. 直接渡されたファイル情報を優先
            if current_files:
                image_files = current_files
                self.logger.debug(f"Using provided file list: {len(image_files)} files")
            
            # 2. バッチデータからの抽出を試行
            if not image_files:
                image_files = self._extract_image_files(batch_data)
                if image_files:
                    self.logger.debug(f"Extracted from batch data: {len(image_files)} files")
                else:
                    self.logger.debug("No files extracted from batch data")
            
            # 履歴に追加
            record = {
                'step': step,
                'loss': loss,
                'image_files': image_files,
                'timestamp': datetime.now().isoformat()
            }
            self.loss_history.append(record)
            
            # ファイル別統計更新
            self._update_file_stats(image_files, loss)
            
            # 閾値チェック
            if loss > self.loss_threshold:
                self._handle_high_loss_alert(loss, step, image_files)
            
            # TensorBoard統合
            if tb_writer:
                self._log_to_tensorboard(tb_writer, loss, step, image_files)
            
            # 定期的なJSON保存
            if step % self.save_frequency == 0:
                self._save_to_json()
                
        except Exception as e:
            self.logger.error(f"Error in log_loss: {e}")
    
    def _extract_image_files(self, batch_data: Union[Dict[str, Any], tuple, Any]) -> List[str]:
        """バッチデータからimage_fileリストを抽出
        
        Args:
            batch_data: 辞書形式またはタプル形式のバッチデータ
            
        Returns:
            抽出されたファイル名のリスト（空リストの場合もあり）
        """
        try:
            if not batch_data:
                self.logger.debug("[LossDebugger] Empty batch_data provided")
                return []
            
            # デバッグ情報を出力
            self.logger.info(f"[LossDebugger] Batch data type: {type(batch_data)}")
            
            # 辞書形式の場合
            if isinstance(batch_data, dict):
                self.logger.info(f"[LossDebugger] Batch data keys: {list(batch_data.keys())}")
                
                # 複数の可能性のあるキー名を試行
                possible_keys = ['image_file', 'image_files', 'file', 'files', 'image_path', 'image_paths']
                
                for key in possible_keys:
                    if key in batch_data:
                        files = batch_data[key]
                        self.logger.info(f"[LossDebugger] Found files under key '{key}': {type(files)}")
                        
                        if isinstance(files, torch.Tensor):
                            # Tensor の場合は文字列に変換
                            try:
                                if files.numel() == 0:
                                    self.logger.info("[LossDebugger] Empty tensor found")
                                    return []
                                result = [str(f) for f in files.tolist()]
                                self.logger.info(f"[LossDebugger] Extracted {len(result)} files from tensor")
                                return result
                            except Exception as tensor_e:
                                self.logger.warning(f"[LossDebugger] Failed to convert tensor: {tensor_e}")
                                return []
                        elif isinstance(files, (list, tuple)):
                            result = [str(f) for f in files if f is not None]
                            self.logger.info(f"[LossDebugger] Extracted {len(result)} files from list/tuple")
                            return result
                        else:
                            result = [str(files)] if files is not None else []
                            self.logger.info(f"[LossDebugger] Extracted single file: {result[0] if result else 'None'}")
                            return result
                
                # ファイル関連のキーが見つからない場合
                self.logger.info(f"[LossDebugger] No file-related keys found in batch data. Available keys: {list(batch_data.keys())}")
                return []
            
            # タプル形式の場合 (features, label)
            elif isinstance(batch_data, tuple):
                self.logger.info(f"[LossDebugger] Tuple batch data with length: {len(batch_data)}")
                
                # タプル形式でも、各要素を再帰的にチェック
                for i, item in enumerate(batch_data):
                    self.logger.debug(f"[LossDebugger] Checking tuple item {i}: {type(item)}")
                    if isinstance(item, dict):
                        files = self._extract_image_files(item)
                        if files:
                            self.logger.info(f"[LossDebugger] Found files in tuple item {i}")
                            return files
                
                self.logger.info("[LossDebugger] Tuple format batch detected. No file information found.")
                return []
            
            # その他の形式
            else:
                self.logger.info(f"[LossDebugger] Unsupported batch data type: {type(batch_data)}")
                return []
            
        except Exception as e:
            self.logger.warning(f"[LossDebugger] Failed to extract image_files: {e}")
            self.logger.debug(f"[LossDebugger] Batch data type: {type(batch_data)}")
            if hasattr(batch_data, 'keys'):
                self.logger.debug(f"[LossDebugger] Keys: {list(batch_data.keys())}")
            elif isinstance(batch_data, (tuple, list)):
                self.logger.debug(f"[LossDebugger] Length: {len(batch_data)}")
            return []
    
    def _update_file_stats(self, image_files: List[str], loss: float) -> None:
        """ファイル別統計を更新"""
        for file_path in image_files:
            if file_path not in self.file_loss_stats:
                self.file_loss_stats[file_path] = {
                    'count': 0,
                    'total_loss': 0.0,
                    'max_loss': 0.0,
                    'min_loss': float('inf'),
                    'recent_losses': deque(maxlen=10)
                }
            
            stats = self.file_loss_stats[file_path]
            stats['count'] += 1
            stats['total_loss'] += loss
            stats['max_loss'] = max(stats['max_loss'], loss)
            stats['min_loss'] = min(stats['min_loss'], loss)
            stats['recent_losses'].append(loss)
    
    def _handle_high_loss_alert(
        self,
        loss: float,
        step: int,
        image_files: List[str]
    ) -> None:
        """高いloss値のアラート処理"""
        # アラート頻度制御
        if step - self.last_alert_step < self.alert_frequency:
            return
        
        self.last_alert_step = step
        
        # アラートメッセージ
        alert_msg = f"HIGH LOSS ALERT - Step {step}: Loss {loss:.4f} (threshold: {self.loss_threshold})"
        if image_files:
            file_names = [Path(f).name for f in image_files[:3]]  # 最初の3ファイルのみ表示
            alert_msg += f" | Files: {', '.join(file_names)}"
            if len(image_files) > 3:
                alert_msg += f" (+{len(image_files)-3} more)"
        
        self.logger.warning(alert_msg)
        ds_logger.warning(alert_msg)
    
    def _log_to_tensorboard(
        self,
        tb_writer: Any,
        loss: float,
        step: int,
        image_files: List[str]
    ) -> None:
        """TensorBoardに追加のデバッグ情報をログ"""
        try:
            # 基本的なloss統計
            if len(self.loss_history) > 1:
                recent_losses = [r['loss'] for r in list(self.loss_history)[-10:]]
                avg_recent_loss = sum(recent_losses) / len(recent_losses)
                tb_writer.add_scalar('debug/recent_avg_loss', avg_recent_loss, step)
            
            # 閾値を超えた回数
            high_loss_count = sum(1 for r in self.loss_history if r['loss'] > self.loss_threshold)
            tb_writer.add_scalar('debug/high_loss_count', high_loss_count, step)
            
            # ファイル別統計のトップ5（最大loss）
            if self.file_loss_stats:
                top_files = sorted(
                    self.file_loss_stats.items(),
                    key=lambda x: x[1]['max_loss'],
                    reverse=True
                )[:5]
                
                for i, (file_path, stats) in enumerate(top_files):
                    file_name = Path(file_path).stem[:20]  # ファイル名を短縮
                    tb_writer.add_scalar(
                        f'debug/top_loss_files/max_loss_{i+1}_{file_name}',
                        stats['max_loss'],
                        step
                    )
                    
        except Exception as e:
            self.logger.warning(f"TensorBoard logging failed: {e}")
    
    def _save_to_json(self) -> None:
        """現在の統計をJSONファイルに保存"""
        if not self.output_dir:
            return
        
        try:
            # 保存データの準備
            export_data = {
                'metadata': {
                    'last_step': self.step_count,
                    'total_records': len(self.loss_history),
                    'loss_threshold': self.loss_threshold,
                    'generated_at': datetime.now().isoformat()
                },
                'recent_history': list(self.loss_history)[-100:],  # 最新100件
                'file_statistics': self._get_file_stats_summary(),
                'high_loss_files': self._get_high_loss_files()
            }
            
            # JSON保存
            with open(self.json_output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
                
            self.logger.debug(f"Debug data saved to {self.json_output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save JSON: {e}")
    
    def _get_file_stats_summary(self) -> Dict[str, Any]:
        """ファイル統計のサマリーを生成"""
        summary = {}
        for file_path, stats in self.file_loss_stats.items():
            avg_loss = stats['total_loss'] / stats['count'] if stats['count'] > 0 else 0
            summary[file_path] = {
                'count': stats['count'],
                'avg_loss': round(avg_loss, 4),
                'max_loss': round(stats['max_loss'], 4),
                'min_loss': round(stats['min_loss'], 4),
                'recent_avg': round(
                    sum(stats['recent_losses']) / len(stats['recent_losses']), 4
                ) if stats['recent_losses'] else 0
            }
        return summary
    
    def _get_high_loss_files(self) -> List[Dict[str, Any]]:
        """高いlossを示すファイルのリストを生成"""
        high_loss_files = []
        for file_path, stats in self.file_loss_stats.items():
            if stats['max_loss'] > self.loss_threshold:
                avg_loss = stats['total_loss'] / stats['count'] if stats['count'] > 0 else 0
                high_loss_files.append({
                    'file_path': file_path,
                    'max_loss': round(stats['max_loss'], 4),
                    'avg_loss': round(avg_loss, 4),
                    'count': stats['count']
                })
        
        # max_lossでソート
        return sorted(high_loss_files, key=lambda x: x['max_loss'], reverse=True)
    
    def get_summary(self) -> Dict[str, Any]:
        """現在の統計サマリーを取得"""
        if not self.enabled:
            return {'enabled': False}
        
        total_files = len(self.file_loss_stats)
        high_loss_files = len(self._get_high_loss_files())
        
        recent_losses = [r['loss'] for r in list(self.loss_history)[-50:]]
        avg_recent_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0
        
        return {
            'enabled': True,
            'total_steps': self.step_count,
            'total_files_processed': total_files,
            'high_loss_files_count': high_loss_files,
            'recent_avg_loss': round(avg_recent_loss, 4),
            'loss_threshold': self.loss_threshold,
            'total_records': len(self.loss_history)
        }
    
    def finalize(self) -> None:
        """トレーニング終了時の最終処理"""
        if not self.enabled:
            return
        
        try:
            # 最終データ保存
            self._save_to_json()
            
            # サマリー出力
            summary = self.get_summary()
            self.logger.info("=== Loss Debugger Final Summary ===")
            for key, value in summary.items():
                self.logger.info(f"{key}: {value}")
            
            if self.output_dir:
                self.logger.info(f"Debug data saved to: {self.json_output_path}")
                
        except Exception as e:
            self.logger.error(f"Error in finalize: {e}")
